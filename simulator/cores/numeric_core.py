#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
from abc import ABCMeta
from . import ICore
from simulator.parameters.core_parameters import CoreStyle
from simulator.circuits.array_simulator import *
from simulator.devices.device import Device
import numpy.typing as npt
import typing

from simulator.backend import ComputeBackend

xp = ComputeBackend()  # Represents either cupy or numpy


class NumericCore(ICore, metaclass=ABCMeta):
    """An inner :py:class:`.ICore` that performs purely-numeric calculations."""

    def __init__(self, params):
        self.matrix = None
        self.vector_vmm = None
        self.vector_mvm = None
        self.par_mask = None
        self.params = params

        # Device and DAC created in numeric core
        # ADC belongs to wrapper core
        self.device = Device.create_device(params.xbar.device)

        if self.params.core.style != CoreStyle.OFFSET:
            self.interleaved = self.params.core.balanced.interleaved_posneg
        else:
            self.interleaved = False

        self.Ncopy = (
            self.params.simulation.convolution.x_par
            * self.params.simulation.convolution.y_par
        )
        self.Rp_row = self.params.xbar.array.parasitics.Rp_row
        self.Rp_col = self.params.xbar.array.parasitics.Rp_col
        self.simulate_parasitics = self.params.xbar.array.parasitics.enable and (
            self.Rp_col > 0 or self.Rp_row > 0
        )

        # Set parasitics solver functions
        # Convention: row_in == True for MVM and row_in = False for VMM
        self.circuit_solver_mvm = None
        self.circuit_solver_vmm = None
        if self.simulate_parasitics:
            if self.interleaved:
                self.circuit_solver_mvm = mvm_parasitics_interleaved
                self.circuit_solver_vmm = mvm_parasitics_interleaved
            else:
                # Note that gate_input is not physically equivalent to Rp=0 on the input side
                # If gate_input = False, the un-activated rows are at 0V
                # If gate_input = True, the un-activated rows are open
                # If gate_input = True, input bit slicing must be on
                # The logic below addresses this
                if self.params.xbar.array.parasitics.gate_input:
                    self.circuit_solver_mvm = mvm_parasitics_gateInput
                    self.circuit_solver_vmm = mvm_parasitics_gateInput
                else:
                    self.circuit_solver_mvm = mvm_parasitics
                    self.circuit_solver_vmm = mvm_parasitics

    def set_matrix(self, matrix, error_mask=None):
        if self.params.simulation.useGPU:
            self.matrix = xp.array(matrix)
        else:
            self.matrix = matrix

        # If simulating parasitics with SW packing, create a mask here
        if self.Ncopy > 1 and self.simulate_parasitics:
            Nx, Ny = matrix.shape
            self.par_mask = xp.zeros(
                (self.Ncopy * Nx, self.Ncopy * Ny),
                dtype=self.matrix.dtype,
            )
            for m in range(self.Ncopy):
                x_start, x_end = m * Nx, (m + 1) * Nx
                y_start, y_end = m * Ny, (m + 1) * Ny
                self.par_mask[x_start:x_end, y_start:y_end] = 1
            self.par_mask = self.par_mask > 1e-9

        # Apply weight error
        matrix_copy = self.matrix.copy()
        matrix_error = self.device.apply_write_error(matrix_copy)
        if not error_mask:
            self.matrix = matrix_error
        else:
            self.matrix = matrix_copy
            self.matrix[error_mask] = matrix_error[error_mask]

    def set_vmm_inputs(self, vector):
        self.vector_vmm = vector

    def set_mvm_inputs(self, vector):
        self.vector_mvm = vector

    def run_xbar_vmm(
        self,
        vector: typing.Optional[npt.NDArray] = None,
        core_neg: "NumericCore" = None,
    ):
        # apply read noise
        matrix = self.read_noise_matrix()
        print(matrix)
        if self.interleaved:
            matrix_neg = core_neg.read_noise_matrix()
        else:
            matrix_neg = None

        if vector is None:
            vector = self.vector_vmm

        circuit_solver = self.circuit_solver_vmm
        row_in = False
        op_pair = (vector, matrix)

        return self.run_xbar_operation(
            matrix,
            vector,
            op_pair,
            circuit_solver,
            row_in,
            matrix_neg,
        )

    def run_xbar_mvm(
        self,
        vector: typing.Optional[npt.NDArray] = None,
        core_neg: "NumericCore" = None,
    ) -> npt.NDArray:
        # Apply read noise (unique noise on each call)
        matrix = self.read_noise_matrix()
        if self.interleaved:
            matrix_neg = core_neg.read_noise_matrix()
        else:
            matrix_neg = None

        # Load input vector
        if vector is None:
            vector = self.vector_mvm

        circuit_solver = self.circuit_solver_mvm
        row_in = True
        op_pair = (matrix, vector)

        return self.run_xbar_operation(
            matrix,
            vector,
            op_pair,
            circuit_solver,
            row_in,
            matrix_neg,
        )

    def run_xbar_operation(
        self,
        matrix,
        vector,
        op_pair,
        circuit_solver,
        row_in,
        matrix_neg,
    ):
        if self.simulate_parasitics and vector.any():
            useMask = self.Ncopy > 1
            result = solve_mvm_circuit(
                circuit_solver,
                vector,
                matrix.copy(),
                self.params,
                interleaved=self.interleaved,
                matrix_neg=matrix_neg,
                useMask=useMask,
                mask=self.par_mask,
                row_in=row_in,
            )

        elif matrix_neg is not None:
            # Interleaved without parasitics: identical to normal balanced core operation
            if row_in:
                result = xp.dot(*op_pair) - xp.dot(matrix_neg, vector)
            else:
                result = xp.dot(*op_pair) - xp.dot(vector, matrix_neg)
        else:
            # Compute using matrix vector dot product
            result = xp.dot(*op_pair)

        return result

    def _read_matrix(self):
        return self.matrix.copy()

    def _save_matrix(self):
        return self.matrix.copy()

    def _restore_matrix(self, matrix):
        self.matrix = matrix.copy()

    def read_noise_matrix(self) -> npt.NDArray:
        """Applies noise to a weight matrix, accounting for whether the matrix inclues replicated weights."""
        # Default code path
        if self.Ncopy == 1:
            return self.device.read_noise(self.matrix.copy())

        # If doing a circuit simulation, must keep the full sized block diagonal matrix
        elif self.Ncopy > 1 and self.simulate_parasitics:
            noisy_matrix = self.device.read_noise(self.matrix.copy())
            noisy_matrix *= self.par_mask

        # If not parasitic and Ncopy > 1
        else:
            if not self.params.xbar.device.read_noise.enable:
                return self.matrix
            else:
                noisy_matrix = self.device.read_noise(self.matrix_dense.copy())
                Nx, Ny = self.matrix_original.shape
                for m in range(self.Ncopy):
                    x_start, y_start = m * Nx, m * Ny
                    x_end, y_end = x_start + Nx, y_start + Ny
                    self.matrix[x_start:x_end, y_start:y_end] = noisy_matrix[m, :, :]
                noisy_matrix = self.matrix

        return noisy_matrix

    def expand_matrix(self, Ncopy):
        """Makes a big matrix containing M copies of the weight matrix so that multiple VMMs can be computed in parallel, SIMD style
        Off-diagonal blocks of this matrix are all zero
        If noise is enabled, additionally create a third matrix that contains all the nonzero elements of this big matrix
        Intended for GPU use only, designed for neural network inference.
        """
        # Keep a copy of original matrix, both for construction of the expanded matrix and as a backup for later restoration if needed

        Nx, Ny = self.matrix.shape

        # Keep a copy of the original un-expanded matrix so that it can be restored with unexpand_matrix
        self.matrix_original = self.matrix.copy()

        if not self.params.xbar.device.read_noise.enable:
            if self.params.simulation.convolution.weight_reorder:
                self.matrix = self.weight_reorder(self.matrix_original.copy())

            else:
                self.matrix = xp.zeros(
                    (Ncopy * Nx, Ncopy * Ny),
                    dtype=self.matrix.dtype,
                )
                for m in range(Ncopy):
                    x_start, x_end = m * Nx, (m + 1) * Nx
                    y_start, y_end = m * Ny, (m + 1) * Ny
                    self.matrix[x_start:x_end, y_start:y_end] = self.matrix_original

        else:
            # Block diagonal matrix for running MVMs
            self.matrix = xp.zeros((Ncopy * Nx, Ncopy * Ny), dtype=self.matrix.dtype)
            # Dense matrix with the same number of non-zeros as the block diagonal for applying read noise
            self.matrix_dense = xp.zeros((Ncopy, Nx, Ny), dtype=self.matrix.dtype)
            for m in range(Ncopy):
                x_start, x_end = m * Nx, (m + 1) * Nx
                y_start, y_end = m * Ny, (m + 1) * Ny
                self.matrix[x_start:x_end, y_start:y_end] = self.matrix_original
                self.matrix_dense[m, :, :] = self.matrix_original

    def weight_reorder(self, matrix_original):
        """Utility function used to implement weight reordering for sliding window packing
        This function is also used by higher cores if fast_balanced = True.
        """
        Kx = self.params.simulation.convolution.Kx
        Ky = self.params.simulation.convolution.Ky
        Nic = self.params.simulation.convolution.Nic
        Noc = self.params.simulation.convolution.Noc
        stride = self.params.simulation.convolution.stride
        x_par = self.params.simulation.convolution.x_par  # parallel windows in x
        y_par = self.params.simulation.convolution.y_par  # parallel windows in y
        x_par_in = (x_par - 1) * stride + Kx
        y_par_in = (y_par - 1) * stride + Ky

        matrix = xp.zeros(
            (x_par * y_par * Noc, x_par_in * y_par_in * Nic),
            dtype=matrix_original.dtype,
        )
        m = 0
        for ix in range(x_par):
            for iy in range(y_par):
                for ixx in range(Kx):
                    for iyy in range(Ky):
                        # 1: Which elements of the flattened input should be indexed for this 2D point?
                        x_coord = stride * ix + ixx
                        y_coord = stride * iy + iyy
                        row_xy = x_coord * y_par_in + y_coord
                        x_start = row_xy
                        x_end = row_xy + Nic * x_par_in * y_par_in
                        # 2: Which elements of the weight matrix are used for this point?
                        Wx_coord = ixx * Ky + iyy
                        W_start = Wx_coord
                        W_end = Wx_coord + Nic * Kx * Ky
                        y_start, y_end = m * Noc, (m + 1) * Noc
                        matrix[
                            y_start:y_end,
                            x_start : x_end : (x_par_in * y_par_in),
                        ] = matrix_original[:, W_start : W_end : (Kx * Ky)].copy()
                m += 1

        return matrix

    def unexpand_matrix(self):
        """Undo the expansion operation in expand_matrix."""
        self.matrix = self.matrix_original.copy()
        self.matrix_dense = None
