#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import time
import numpy as np
import onnxruntime as ort
import onnx
import os
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
import torch
import torch.nn.functional as F
from scipy.spatial import distance
from skimage.metrics import structural_similarity as compare_ssim
from scipy.spatial.distance import pdist, squareform
import keras
import keras.backend as K
import math
from statistics import geometric_mean

from .activate import Activate, RECTLINEAR, STYLES
from .convolution import Convolution
from .dnn_util import (
    apply_pool,
    flatten_layer,
    reducemean_layer,
    space_to_depth,
    apply_quantization,
    apply_quantization_onnx,
    apply_dequantization_onnx,
    init_GPU_util,
    decode_from_key,
)
from ...cores.analog_core import AnalogCore
from ...parameters.core_parameters import CoreStyle, BitSlicedCoreStyle
from ...backend import ComputeBackend

xp = ComputeBackend()


class DNN:
    """This class creates a multi-layer neural network, reads an input data set, and
    uses the network to perform classification with pre-trained weights.
    """

    def __init__(self, layers, seed=None):
        """Define a neural network object
        required args:
          layers defined for a convolutional network as a tuple of tuples  (or list of lists). each inner tuple represents: (x size, y size, channels)
                Thus we have:( (x_in,y_in,channel_in),(x_hidden1,y_hidden1,channel_hidden1), ...,(x_out,y_out,channel_out))
                fully connected layers should be specified as:
                    ( (1,1,n_in),(1,1,n_hidden1), ...,(1,1,n_out)) OR ( n_in,n_hidden1, ...,n_out).
        """
        self.layers = []
        for layer in layers:
            if type(layer) == int:
                self.layers.append((1, 1, layer))
            else:
                self.layers.append(layer)

        self.nlayer = len(layers) - 1
        self.ndata = 0
        self.indata = None
        self.answers = None
        self.layerTypes = None

        # default activation function, redefine using set_activations
        self.activations = [None for k in range(self.nlayer)]
        for k in range(self.nlayer):
            self.activations[k] = Activate(style="SIGMOID")

        # Batch norm
        self.batch_norm = [False for layer in range(self.nlayer)]
        self.batch_norm_params = [None for layer in range(self.nlayer)]
        self.batch_norm_epsilons = [1e-3 for layer in range(self.nlayer)]

        # Parameters for non MVM active layers (pool, add, concat)
        self.auxLayerParams = self.nlayer * [None]

        # ID of the layer that sources each layer (necessary for ResNets, etc)
        self.sourceLayers = self.nlayer * [None]
        self.memory_window = 0

        # cores are defined by call to ncore()
        self.ncore_style = self.nlayer * [None]  # create list of length nlayer
        self.bias_row = self.nlayer * [False]
        self.ncores = self.nlayer * [None]  # create list of length nlayer

        self.whetstone = False
        self.decoder_mat = None
        self.digital_bias = self.nlayer * [False]
        self.bias_values = self.nlayer * [None]
        self.scale_values = self.nlayer * [None]
        self.quantization_values = self.nlayer * [None]
        self.useGPU = False
        self.gpu_id = 0
        self.profile_DAC_inputs = False
        self.profile_ADC_inputs = False
        self.profile_ADC_reluAware = False

        # Set CPU as default for utility functions
        init_GPU_util(False)

        # initialize Numpy and Python RNGs
        if seed is not None:
            np.random.seed(seed)

    # -------------------------------------------------------
    # Set parameters that are used for inference
    # -------------------------------------------------------
    # def gram_matrix(self,X):
    #     X_flattened = X.reshape(X.shape[0], -1) # X.shape[0]은 샘플의 수, -1은 나머지 차원을 평탄화
    #     return X_flattened @ X_flattened.T

    # def centered_kernel_alignment(self, X, Y):
    #     X = X - X.mean(axis=0)
    #     Y = Y - Y.mean(axis=0)

    #     KX = self.gram_matrix(X)
    #     KY = self.gram_matrix(Y)

    #     return np.trace(KX @ KY) / (np.sqrt(np.trace(KX @ KX)) * np.sqrt(np.trace(KY @ KY)))
    

    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n

        return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
        # return np.dot(H, K)  # KH


    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX


    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))


    def linear_HSIC(self, X, Y):
        L_X = np.dot(X, X.T)
        L_Y = np.dot(Y, Y.T)
        return np.sum(self.centering(L_X) * self.centering(L_Y))


    def linear_CKA(self,X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)


    def kernel_CKA(self,X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)


    def cosine_similarity(self, tensor_a, tensor_b):
        # Flatten the tensors to 1D vectors
        vec_a = tensor_a.flatten()
        vec_b = tensor_b.flatten()

        # Compute the dot product of the vectors
        dot_product = np.dot(vec_a, vec_b)

        # Compute the magnitude (norm) of each vector
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        # Compute the cosine similarity
        similarity = dot_product / (norm_a * norm_b)
        
        return similarity
    def hamming_distance(self, array_a, array_b):
        # 배열의 모양을 확인하여 같은지 확인합니다.
        if array_a.shape != array_b.shape:
            raise ValueError("Arrays must have the same shape")

        # 두 배열 사이의 차이를 계산합니다.
        return np.sum(array_a != array_b)
    
    def binary_transform(self, array):
        return np.where(array > 0, 1, 0)
    def set_inference_params(
        self,
        layerParams,
        memory_window=1,
        whetstone=False,
        batchnorm_style="sqrt",
        fold_batchnorm=False,
    ):
        if len(layerParams) != self.nlayer:
            raise ValueError(
                "Number of layers defined in layerParams is inconstent with sizes vector",
            )
        self.layerTypes = [layerParams[k]["type"] for k in range(len(layerParams))]
        self.sourceLayers = [layerParams[k]["source"] for k in range(len(layerParams))]
        self.memory_window = memory_window
        self.whetstone = whetstone
        self.batchnorm_style = batchnorm_style
        self.fold_batchnorm = fold_batchnorm

    def set_layer_params(self, ilayer, layerParam, digital_bias):
        self.auxLayerParams[ilayer] = layerParam
        if (
            layerParam["type"] in ("conv", "dense")
            and layerParam["bias"]
            and not digital_bias
        ):
            self.bias_row[ilayer] = True
        if (
            layerParam["type"] in ("conv", "dense")
            and digital_bias
            and (
                layerParam["bias"]
                or (self.fold_batchnorm and layerParam["batch_norm"] is not None)
            )
        ):
            self.digital_bias[ilayer] = True
        if layerParam["batch_norm"] is None:
            self.batch_norm[ilayer] = False
        else:
            if self.layerTypes[ilayer] in ("conv", "dense") and self.fold_batchnorm:
                self.batch_norm[ilayer] = False
            else:
                self.batch_norm[ilayer] = True

    # -------------------------------------------------------
    # Define neural core and activation
    # -------------------------------------------------------

    # create one or more neural cores
    # style = "conv", "dense" (for inference)
    # which = 0 to create all cores
    # which = 1 to N to create Nth core
    def ncore(self, which, style="dense", **kwargs):
        self.ncore_style[which] = style
        self.ncores[which] = self.create_onecore(style, which, **kwargs)

    def create_onecore(self, style, i, **kwargs):
        if style == "conv":
            # Create a convolutional layer
            # The row and column numbers here will be ignored: an appropriately sized matrix
            # will be initialized based on convolutional layer parameters (params)
            ncore = Convolution(self.auxLayerParams[i], **kwargs)
        elif style == "dense":
            # If core is not a conv layer, need to check that specified inputs and outputs are indeed vectors
            if self.layers[i][0] != 1 and self.layers[i][1] != 1:
                raise ValueError("Invalid size of input to a fully connected layer")
            if self.layers[i + 1][0] != 1 and self.layers[i + 1][1] != 1:
                raise ValueError("Invalid size of output from a fully connected layer")
            if not self.bias_row[i]:
                ncore = AnalogCore(
                    np.zeros((self.layers[i + 1][2], self.layers[i][2])),
                    empty_matrix=True,
                    **kwargs,
                )
            else:
                ncore = AnalogCore(
                    np.zeros((self.layers[i + 1][2], self.layers[i][2] + 1)),
                    empty_matrix=True,
                    **kwargs,
                )
        else:
            error("Unknown neural core style")

        return ncore

    def set_activations(self, layer, **kwargs):
        self.activations[layer] = Activate(**kwargs, useGPU=self.useGPU)

    # -------------------------------------------------------
    # Load weights from Keras weight dictionary
    # -------------------------------------------------------

    def read_weights_keras(self, weight_dict, verbose=False):
        """Populate Ncore weight matrices from a Keras model file
        weight_dict: dictionary container of weights from Keras model indexed by layer name.
        """
        for ilayer in range(self.nlayer):
            # Extract batch norm if applicable
            if self.batch_norm[ilayer]:
                Wbn = weight_dict[self.auxLayerParams[ilayer]["batch_norm"]]
                if (
                    self.auxLayerParams[ilayer]["BN_scale"]
                    and not self.auxLayerParams[ilayer]["BN_center"]
                ):
                    Wbn = [Wbn[0], np.zeros(Wbn[0].shape), Wbn[1], Wbn[2]]
                elif (
                    not self.auxLayerParams[ilayer]["BN_scale"]
                    and self.auxLayerParams[ilayer]["BN_center"]
                ):
                    Wbn = [np.ones(Wbn[0].shape), Wbn[0], Wbn[1], Wbn[2]]
                elif (
                    not self.auxLayerParams[ilayer]["BN_scale"]
                    and not self.auxLayerParams[ilayer]["BN_center"]
                ):
                    Wbn = [
                        np.ones(Wbn[0].shape),
                        np.zeros(Wbn[0].shape),
                        Wbn[0],
                        Wbn[1],
                    ]
                for j in range(len(Wbn)):
                    Wbn[j] = Wbn[j].astype(xp.float32)
                    if self.layerTypes[ilayer] in ("conv", "pool"):
                        Wbn[j] = Wbn[j].reshape((len(Wbn[j]), 1, 1))
                    Wbn[j] = xp.array(Wbn[j])
                self.batch_norm_epsilons[ilayer] = self.auxLayerParams[ilayer][
                    "epsilon"
                ]
                self.batch_norm_params[ilayer] = Wbn

            if self.layerTypes[ilayer] not in ("conv", "dense"):
                continue

            # Extract raw tensors from Keras model
            Wi_0 = weight_dict[self.auxLayerParams[ilayer]["name"]]
            Wi = Wi_0[0].astype(xp.float32)

            # Extract the bias weights
            if self.bias_row[ilayer]:
                useBias = True
                Wbias = Wi_0[1].astype(xp.float32)
            else:
                useBias = False

            matrix = np.zeros(
                (self.ncores[ilayer].nrow, self.ncores[ilayer].ncol), dtype=Wi.dtype,
            )

            if self.ncore_style[ilayer] == "conv":
                Noc,Nic,Kx, Ky = Wi.shape
                if self.auxLayerParams[ilayer]["depthwise"]:
                    Nic = Noc

                # Check dimensions
                if Noc != matrix.shape[0]:
                    raise ValueError("Mismatch in conv layer along output dimension")
                if Kx != self.auxLayerParams[ilayer]["Kx"]:
                    raise ValueError("Mismatch in kernel size Kx")
                if Ky != self.auxLayerParams[ilayer]["Ky"]:
                    raise ValueError("Mismatch in kernel size Kx")
                if Nic != self.auxLayerParams[ilayer]["Nic"]:
                    raise ValueError(
                        "Mismatch in input channel size: "
                        + str(Nic)
                        + " vs "
                        + str(self.auxLayerParams[ilayer]["Nic"]),
                    )
                if (useBias and int(Kx * Ky * Nic + 1) != matrix.shape[1]) or (
                    not useBias and int(Kx * Ky * Nic) != matrix.shape[1]
                ):
                    raise ValueError("Mismatch in conv layer along input dimension")

                if not self.auxLayerParams[ilayer]["depthwise"]:
                    for i in range(Noc):
                        submat = np.array(
                            [Wi[i, k, :, :].flatten() for k in range(Nic)],
                        ).flatten()
                        if useBias:
                            matrix[i, :-1] = submat
                        else:
                            matrix[i, :] = submat
                    if useBias:
                        matrix[:, -1] = Wbias

                else:
                    for i in range(Noc):
                        matrix[i, (i * Kx * Ky) : ((i + 1) * Kx * Ky)] = Wi[
                            i, 0, :, :,
                        ].flatten()
                    if useBias:
                        matrix[:, -1] = Wbias
            else:
                # Wi = Wi.transpose()
                if not useBias:
                    if matrix.shape != Wi.shape:
                        raise ValueError("Mismatch in FC layer dimensions (no bias)")
                    matrix = Wi
                elif useBias:
                    if (matrix.shape[0] != Wi.shape[0]) or (
                        matrix.shape[1] != (Wi.shape[1] + 1)
                    ):
                        raise ValueError("Mismatch in FC layer dimensions (with bias)")
                    matrix[:, :-1] = Wi
                    matrix[:, -1] = Wbias

            self.ncores[ilayer].set_matrix(matrix, verbose=verbose)

        # Whetstone decoder mat
        if self.whetstone:
            last_key = list(weight_dict)[-1]
            self.decoder_mat = weight_dict[last_key][0]

    def read_weights_keras_one_layer(self, weight_dict,layer_num, verbose=False):
        """Populate Ncore weight matrices from a Keras model file
        weight_dict: dictionary container of weights from Keras model indexed by layer name.
        """
        ilayer = layer_num
            # Extract batch norm if applicable
        if self.batch_norm[ilayer]:
            Wbn = weight_dict[self.auxLayerParams[ilayer]["batch_norm"]]
            if (
                self.auxLayerParams[ilayer]["BN_scale"]
                and not self.auxLayerParams[ilayer]["BN_center"]
            ):
                Wbn = [Wbn[0], np.zeros(Wbn[0].shape), Wbn[1], Wbn[2]]
            elif (
                not self.auxLayerParams[ilayer]["BN_scale"]
                and self.auxLayerParams[ilayer]["BN_center"]
            ):
                Wbn = [np.ones(Wbn[0].shape), Wbn[0], Wbn[1], Wbn[2]]
            elif (
                not self.auxLayerParams[ilayer]["BN_scale"]
                and not self.auxLayerParams[ilayer]["BN_center"]
            ):
                Wbn = [
                    np.ones(Wbn[0].shape),
                    np.zeros(Wbn[0].shape),
                    Wbn[0],
                    Wbn[1],
                ]
            for j in range(len(Wbn)):
                Wbn[j] = Wbn[j].astype(xp.float32)
                if self.layerTypes[ilayer] in ("conv", "pool"):
                    Wbn[j] = Wbn[j].reshape((len(Wbn[j]), 1, 1))
                Wbn[j] = xp.array(Wbn[j])
            self.batch_norm_epsilons[ilayer] = self.auxLayerParams[ilayer][
                "epsilon"
            ]
            self.batch_norm_params[ilayer] = Wbn



        # Extract raw tensors from Keras model
        Wi_0 = weight_dict[self.auxLayerParams[ilayer]["name"]]
        Wi = Wi_0[0].astype(xp.float32)

        # Extract the bias weights
        if self.bias_row[ilayer]:
            useBias = True
            Wbias = Wi_0[1].astype(xp.float32)
        else:
            useBias = False

        matrix = np.zeros(
            (self.ncores[ilayer].nrow, self.ncores[ilayer].ncol), dtype=Wi.dtype,
        )

        if self.ncore_style[ilayer] == "conv":
            Noc,Nic,Kx, Ky = Wi.shape
            if self.auxLayerParams[ilayer]["depthwise"]:
                Nic = Noc

            # Check dimensions
            if Noc != matrix.shape[0]:
                raise ValueError("Mismatch in conv layer along output dimension")
            if Kx != self.auxLayerParams[ilayer]["Kx"]:
                raise ValueError("Mismatch in kernel size Kx")
            if Ky != self.auxLayerParams[ilayer]["Ky"]:
                raise ValueError("Mismatch in kernel size Kx")
            if Nic != self.auxLayerParams[ilayer]["Nic"]:
                raise ValueError(
                    "Mismatch in input channel size: "
                    + str(Nic)
                    + " vs "
                    + str(self.auxLayerParams[ilayer]["Nic"]),
                )
            if (useBias and int(Kx * Ky * Nic + 1) != matrix.shape[1]) or (
                not useBias and int(Kx * Ky * Nic) != matrix.shape[1]
            ):
                raise ValueError("Mismatch in conv layer along input dimension")

            if not self.auxLayerParams[ilayer]["depthwise"]:
                for i in range(Noc):
                    submat = np.array(
                        [Wi[i, k, :, :].flatten() for k in range(Nic)],
                    ).flatten()
                    if useBias:
                        matrix[i, :-1] = submat
                    else:
                        matrix[i, :] = submat
                if useBias:
                    matrix[:, -1] = Wbias

            else:
                for i in range(Noc):
                    matrix[i, (i * Kx * Ky) : ((i + 1) * Kx * Ky)] = Wi[
                        i, 0, :, :,
                    ].flatten()
                if useBias:
                    matrix[:, -1] = Wbias
        else:
            # Wi = Wi.transpose()
            if not useBias:
                if matrix.shape != Wi.shape:
                    print(matrix.shape)
                    print(Wi.shape)
                    raise ValueError("Mismatch in FC layer dimensions (no bias)")
                matrix = Wi
            elif useBias:
                if (matrix.shape[0] != Wi.shape[0]) or (
                    matrix.shape[1] != (Wi.shape[1] + 1)
                ):
                
                    raise ValueError("Mismatch in FC layer dimensions (with bias)")
                matrix[:, :-1] = Wi
                matrix[:, -1] = Wbias

        self.ncores[ilayer].set_matrix(matrix, verbose=verbose)



    # -------------------------------------------------------
    # GPU initialization
    # -------------------------------------------------------

    def init_GPU(self, useGPU, gpu_id):
        self.useGPU = useGPU
        if useGPU:
            self.gpu_id = gpu_id
            import cupy

            cupy.cuda.Device(gpu_id).use()
        init_GPU_util(useGPU)

    def expand_cores(self):
        """Duplicate the arrays inside the neural cores in order to allow parallel sliding window computation of convolutions
        Duplication factor is set by x_par and y_par.
        """
        for ilayer in range(self.nlayer):
            if self.layerTypes[ilayer] == "conv":
                Ncopy = (
                    self.ncores[ilayer].params.simulation.convolution.x_par
                    * self.ncores[ilayer].params.simulation.convolution.y_par
                )
                for j in range(self.ncores[ilayer].core.num_cores_row):
                    for k in range(self.ncores[ilayer].core.num_cores_col):
                        self.ncores[ilayer].cores[j][k].expand_matrix(Ncopy)

    def unexpand_cores(self):
        """Undo expand_cores() to free up memory."""
        for ilayer in range(self.nlayer):
            if self.layerTypes[ilayer] == "conv" and (
                self.ncores[ilayer].params.simulation.convolution.x_par > 1
                or self.ncores[ilayer].params.simulation.convolution.y_par > 1
            ):
                for j in range(self.ncores[ilayer].core.num_cores_row):
                    for k in range(self.ncores[ilayer].core.num_cores_col):
                        self.ncores[ilayer].cores[j][k].unexpand_matrix()

    def show_HW_config(self):
        """Show critical information about each core."""
        Ndevices = 0
        Narrays_all = 0
        Nrows_max_util = 0
        Ncols_max_util = 0
        Nmvms_all = 0
        for ilayer in range(self.nlayer):
            print("Layer " + str(ilayer) + ": " + self.auxLayerParams[ilayer]["name"])
            if self.layerTypes[ilayer] == "conv" or self.layerTypes[ilayer] == "dense":
                core_m = self.ncores[ilayer]
                if self.layerTypes[ilayer] == "conv":
                    core_m = core_m.core

                Ncores = core_m.Ncores
                Nslices = 1
                if core_m.params.core.style == CoreStyle.BITSLICED:
                    Nslices = core_m.params.core.bit_sliced.num_slices

                Narrays = Ncores * Nslices
                if core_m.params.core.style == CoreStyle.BALANCED or (
                    core_m.params.core.style == CoreStyle.BITSLICED
                    and core_m.params.core.bit_sliced.style
                    == BitSlicedCoreStyle.BALANCED
                ):
                    Narrays *= 2

                if self.layerTypes[ilayer] == "conv":
                    print(
                        "        Kernel dims: {:d} x {:d} x {:d} x {:d}".format(
                            self.ncores[ilayer].Kx,
                            self.ncores[ilayer].Ky,
                            self.ncores[ilayer].Nic,
                            self.ncores[ilayer].Noc,
                        ),
                    )
                    N_mvms = self.ncores[ilayer].Nox * self.ncores[ilayer].Noy
                    N_mvms *= Narrays
                else:
                    N_mvms = Narrays
                Nmvms_all += N_mvms

                print("           Bias row: " + str(self.bias_row[ilayer]))
                print(
                    "        Matrix dims: {:d} x {:d}".format(core_m.ncol, core_m.nrow),
                )
                print(
                    "   Target bits/cell: {:d}b".format(
                        core_m.params.xbar.device.cell_bits,
                    ),
                )
                print("       # partitions: {:d}".format(Ncores))
                print("       # MVMs/image: {:d}".format(N_mvms))

                Ndevices_m = 0
                for i in range(core_m.num_cores_row):
                    for j in range(core_m.num_cores_col):
                        if Ncores > 1:
                            print(
                                "          partition ({:d}, {:d}): {:d} x {:d}".format(
                                    j, i, core_m.NcolsVec[j], core_m.NrowsVec[i],
                                ),
                            )
                        Ndevices_m += (
                            core_m.NcolsVec[j] * core_m.NrowsVec[i] * (Narrays / Ncores)
                        )
                        if core_m.NcolsVec[j] > Nrows_max_util:
                            Nrows_max_util = core_m.NcolsVec[j]
                        if core_m.NrowsVec[i] > Ncols_max_util:
                            Ncols_max_util = core_m.NrowsVec[i]
                print("       # bit slices: {:d}".format(Nslices))
                print("           # arrays: {:d}".format(Narrays))
                if Ndevices_m > 1e6:
                    print("          # devices: {:.2f}M".format(Ndevices_m / 1e6))
                elif Ndevices_m > 1e3:
                    print("          # devices: {:.1f}K".format(Ndevices_m / 1e3))
                else:
                    print("          # devices: {:d}".format(int(Ndevices_m)))
                Ndevices += Ndevices_m
                Narrays_all += Narrays
            print("--")

        print("======")
        print("Full systems statistics")
        print("     # arrays: {:d}".format(Narrays_all))
        print("   max # rows: {:d}".format(Nrows_max_util))
        print("   max # cols: {:d}".format(Ncols_max_util))
        if Ndevices > 1e6:
            print("    # devices: {:.2f}M".format(Ndevices / 1e6))
        elif Ndevices > 1e3:
            print("    # devices: {:.1f}K".format(Ndevices / 1e3))
        else:
            print("    # devices: {:d}".format(int(Ndevices)))

        if Nmvms_all > 1e3:
            print(" # MVMs/image: {:.1f}K".format(Nmvms_all / 1e3))
        else:
            print(" # MVMs/image: {:d}".format(int(Nmvms_all)))

    # -------------------------------------------------------
    # Classification
    # -------------------------------------------------------

    def predict(
        self,
        n=0,
        count_interval=0,
        randomSampling=True,
        topk=1,
        time_interval=False,
        return_network_output=False,
        profiling_folder=None,
        profiling_settings=[False, False, False],
        weight_dict = None,
        search_accuracy = False,
        search_accuracy_layer = None,
        config = None,
        ort_output_list = None,
        search_accuracy_output_vec_list = None,
        search_accuracy_output_vecs_list = None,
        search_accuracy_output_vecs_add_list = None,
        from_n = None,
        metric = None,
    ):
        """Perform a forward calculation on the dataset
        n = # of inputs to process, 0 = default = all
        Return (# correct predictions, fraction correct predictions).
        """
        if not self.ncore_style:
            raise ValueError("Neural core is not initialized")
        if not n:
            n = self.ndata
        if n > self.ndata:
            warn("N too large for classification")
            n = self.ndata

        # DAC inputs are always profiled inside DNN in algorithmic units
        # ADC inputs are profiled inside DNN only if ReLU-aware profiling is used
        (
            self.profile_DAC_inputs,
            self.profile_ADC_inputs,
            self.profile_ADC_reluAware,
        ) = profiling_settings
        if self.profile_DAC_inputs or self.profile_ADC_reluAware:
            profile_in_dnn = True
            self.profiled_values = [
                xp.zeros(0, dtype=xp.float32) for i in range(self.nlayer)
            ]
        else:
            profile_in_dnn = False
        if self.profile_DAC_inputs or self.profile_ADC_inputs:
            if profiling_folder is None:
                raise ValueError(
                    "Please provide a valid folder to save profiling results to",
                )
        if self.profile_DAC_inputs and self.profile_ADC_inputs:
            raise ValueError(
                "Cannot profile both DAC and ADC statistics simultaneously. Please profile DAC first, then enable the DAC and profile ADC.",
            )

        # Generate a list of random example numbers
        if n == self.ndata:
            randomSampling = False
        if randomSampling:
            print("Randomized data set")
            inds_rand = np.random.choice(np.arange(self.ndata), size=n, replace=False)

        # Check if network branches, e.g. ResNet
        # This is true if sourceLayers has been set
        branch = self.sourceLayers != self.nlayer * [None]
        if time_interval:
            t_avg, one_T, T1 = 0, 0, time.time()

        if not return_network_output:
            network_outputs = None

        count = 0 if type(topk) is int else np.zeros(len(topk), dtype=np.int32)

        # Make predictions on the inputs
        if search_accuracy:
            mse_list=[] 
            if ort_output_list == None:
                ort_output_list = {}
                print("ort_output_list is None")
            if search_accuracy_output_vec_list == None:
                search_accuracy_output_vec_list = {}
                search_accuracy_output_vecs_list = {}
                search_accuracy_output_vecs_add_list = {}   
            for one in range(n):
                ex = one if not randomSampling else inds_rand[one]
                mse, output_ort_copied , output_vec_copied, output_vecs_copied, output_vecs_add_copied ,new_from_n= self.predict_one(
                    ex,
                    branch,
                    topk=topk,
                    profile_in_dnn=profile_in_dnn,
                    return_network_output=return_network_output,
                    weight_dict=weight_dict,
                    search_accuracy=search_accuracy,
                    search_accuracy_layer = search_accuracy_layer,
                    config = config,
                    ort_output = ort_output_list[one] if one in ort_output_list else None,
                    search_accuracy_output_vec = search_accuracy_output_vec_list[one] if one in search_accuracy_output_vec_list else None,
                    search_accuracy_output_vecs = search_accuracy_output_vecs_list[one] if one in search_accuracy_output_vecs_list else None,
                    search_accuracy_output_vecs_add = search_accuracy_output_vecs_add_list[one] if one in search_accuracy_output_vecs_add_list else None,
                    from_n= from_n,
                    metric = metric
                )

                if (one in ort_output_list) is False:
                    ort_output_list[one] = output_ort_copied
                search_accuracy_output_vec_list[one] = output_vec_copied
                search_accuracy_output_vecs_list[one] = output_vecs_copied
                search_accuracy_output_vecs_add_list[one] = output_vecs_add_copied
                mse_list.append(mse)

            # mse_mean = np.mean(mse_list)
            # geometric mean

          
            mse_np = np.array(mse_list)
            # mse_mean = np.median(mse_np)

            mse_mean = mse_np.prod()**(1.0/len(mse_np))
            if metric == "cka":
                mse_mean = -1 * mse_mean
            print("mse_geomean: ", mse_mean)

            tmp_search_accuracy_output_vec_list = {}
            tmp_search_accuracy_output_vecs_list = {}
            tmp_search_accuracy_output_vecs_add_list = {}
            for key in search_accuracy_output_vec_list.keys():
                tmp_search_accuracy_output_vec_list[key] = search_accuracy_output_vec_list[key].copy()
                tmp_search_accuracy_output_vecs_list[key] = search_accuracy_output_vecs_list[key].copy()
                tmp_search_accuracy_output_vecs_add_list[key] = search_accuracy_output_vecs_add_list[key].copy()
            return mse_mean.copy(), ort_output_list, tmp_search_accuracy_output_vec_list.copy(), tmp_search_accuracy_output_vecs_list.copy(), tmp_search_accuracy_output_vecs_add_list.copy(), new_from_n
            

        for one in range(n):
            ### Display the cumulative accuracy and compute time to the user
            if one > 0 and (count_interval > 0) and one % count_interval == 0:
                time_msg, accs = "", ""
                if time_interval:
                    t_avg = t_avg * (one_T / (one_T + 1)) + (time.time() - T1) / (
                        one_T + 1
                    )
                    time_msg = ", time = {:.4f}".format(t_avg) + "s"
                    T1, one_T = time.time(), one_T + 1
                if type(topk) is int:
                    accs = "{:.2f}".format(100 * float(count) / one) + "%"
                else:
                    for j in range(len(topk)):
                        accs += (
                            "{:.2f}".format(100 * float(count[j]) / one)
                            + "% (top-"
                            + str(topk[j])
                            + ")"
                        )
                        if j < (len(topk) - 1):
                            accs += ", "
                print(
                    "Example "
                    + str(one)
                    + "/"
                    + str(n)
                    + ", accuracy so far = "
                    + accs
                    + time_msg,
                    end="\r",
                )

            ### Make a single prediction with the neural network
            ex = one if not randomSampling else inds_rand[one]

            result, output = self.predict_one(
                ex,
                branch,
                topk=topk,
                profile_in_dnn=profile_in_dnn,
                return_network_output=return_network_output,
                weight_dict=weight_dict,
                search_accuracy=search_accuracy,
            )

            ### Accumulate accuracy
            if type(topk) is int:
                count += int(result)
            else:
                for j in range(len(topk)):
                    count[j] += int(result[j])

            ### Collect network outputs
            if return_network_output:
                if one == 0:
                    network_outputs = xp.zeros((n, len(output)))
                network_outputs[one, :] = output

        if count_interval > 0:
            print("\n")

        if return_network_output and self.useGPU:
            network_outputs = network_outputs.get()

        # Save profiled DAC and ADC inputs
        self.save_profiled_data(profiling_folder, profile_in_dnn)

        return count, count / n, network_outputs

    def predict_one(
        self,
        one,
        branch,
        profile_in_dnn=False,
        topk=1,
        debug_graph=False,
        return_network_output=False,
        weight_dict=None,
        search_accuracy = False,
        search_accuracy_layer = None,
        config = None,
        ort_output =None,
        search_accuracy_output_vec = None,
        search_accuracy_output_vecs = None,
        search_accuracy_output_vecs_add = None,
        from_n = None,
        metric = None
    ):
        """Perform a forward calculation on a single input."""
        if one < 0 or one >= self.ndata:
            error("Input index vector %d is out-of-bounds" % one)
        indata = self.indata
        answers = self.answers
        nlayer = self.nlayer
        ncores = self.ncores

        if branch and not search_accuracy:
            sourceLayers = self.sourceLayers
            output_vecs = self.nlayer * [None]
            output_vecs_add = self.nlayer * [None]
            memory_window = (
                self.memory_window
            )  # number of layers to hold in memory before erasure

        elif branch and search_accuracy:
            if search_accuracy_layer == 0:
                ort_output= [None]
                sourceLayers = self.sourceLayers
                output_vecs = self.nlayer * [None]
                output_vecs_add = self.nlayer * [None]
                memory_window = (
                    self.memory_window
                ) 
        
            else :
                ort_output.append(None)
                sourceLayers = self.sourceLayers
                output_vec = search_accuracy_output_vec
                output_vecs = search_accuracy_output_vecs
                output_vecs_add = search_accuracy_output_vecs_add
                memory_window = (
                    self.memory_window
                ) 
            

        for m in range(nlayer):
            if search_accuracy and (from_n is not None) and m <= from_n:
                continue
            if branch and debug_graph:
                print("Layer " + str(m) + ": " + self.layerTypes[m])
                if m > 0:
                    print("   Source 1: layer " + str(sourceLayers[m][0]))
                    if len(sourceLayers[m]) > 1:
                        print("   Source 2: layer " + str(sourceLayers[m][1]))
                print("   Batchnorm: " + str(self.batch_norm[m]))
                print("   Activation: " + str(STYLES[self.activations[m].style]))

            if m == 0:
                ivec = indata[one].copy()
                ort_input = indata[one].copy()
            elif not branch:
                ivec = output_vec.copy()

            #########################################
            #### Convolution and dense layers
            #########################################
            if self.layerTypes[m] in ("conv", "dense"):
                if branch and m != 0:
                    m_src = sourceLayers[m][0]
                    if m_src == -1 :
                        ivec = indata[one].copy()
                        ort_input = indata[one].copy()
                    else:
                        try:
                            if (
                                (self.layerTypes[m_src] == "add" or self.layerTypes[m_src] == "mul") 
                                and self.auxLayerParams[m]["splitBeforeBN"]
                            ):
                                ivec = output_vecs_add[m_src].copy()
                            else:
                                ivec = output_vecs[m_src].copy()
                        except AttributeError:
                            raise ValueError(
                                "Insufficient memory window for the neural network",
                            )

                # Shape input to be compatible with a conv layer, if possibl
                if 'x_zero_point' in self.auxLayerParams[m] and self.auxLayerParams[m]['activation'] is not None:
                    origin_ivec = ivec
                    ivec = (ivec - self.auxLayerParams[m]['x_zero_point'])#.astype(np.uint8)
                    
                search_accuracy_depthwise = False
                if self.layerTypes[m] == "conv":
                    if self.auxLayerParams[m]['depthwise']:
                        wm = weight_dict[self.auxLayerParams[m]["name"]][0]
                        wm_tensor = torch.from_numpy(wm).float() 
                        input_tensor = torch.from_numpy(ivec).unsqueeze(0).float()
                        output = F.conv2d(input_tensor, wm_tensor,padding=self.auxLayerParams[m]["px_0"], groups=self.auxLayerParams[m]["group"], stride=self.auxLayerParams[m]["stride"])
                        mvec = output.squeeze(0).numpy()
                        search_accuracy_depthwise = True  
                        # 이거 지워야 depthwise cpu가 함
                        # mvec = ncores[m].apply_convolution(ivec)                
                    else:
                        mvec = ncores[m].apply_convolution(ivec)
                        if search_accuracy and ort_output[search_accuracy_layer] is None:
                            wm = weight_dict[self.auxLayerParams[m]["name"]][0]
                            wm_tensor = torch.from_numpy(wm).float() 
                            input_tensor = torch.from_numpy(ivec).unsqueeze(0).float()
                            output = F.conv2d(input_tensor, wm_tensor,padding=self.auxLayerParams[m]["px_0"], stride=self.auxLayerParams[m]["stride"])
                            mvec_cpu = output.squeeze(0).numpy()
                        

                elif self.layerTypes[m] == "dense":
                    if self.bias_row[m]:
                        ivec = xp.append(ivec, 1)
                    mvec = ncores[m].matvec(ivec)
                    if search_accuracy and ort_output[search_accuracy_layer] is None:
                        wm = weight_dict[self.auxLayerParams[m]["name"]][0]
                        wm_tensor = torch.from_numpy(wm).float() 
                        input_tensor = torch.from_numpy(ivec).unsqueeze(0).float()
                        output = F.linear(input_tensor, wm_tensor)
                        mvec_cpu = output.squeeze(0).numpy()
         
                if profile_in_dnn:
                    self.update_adc_dac_statistics(m, ivec, mvec)

                # Apply digital bias
                # if m == 5:
                #         print(ivec)
                #         print(mvec)          
                if self.digital_bias[m]:
                    mvec += self.bias_values[m] 
                    if search_accuracy and ort_output[search_accuracy_layer] is None and not search_accuracy_depthwise:
                        mvec_cpu += self.bias_values[m]
                # Apply quant
              

                # Apply batch normalization
                if self.batch_norm[m]:
                    epsilon = self.batch_norm_epsilons[m]
                    gamma, beta, mu, var = self.batch_norm_params[m]
                    if self.batchnorm_style == "sqrt":
                        mvec = gamma * (mvec - mu) / xp.sqrt(var + epsilon) + beta
                    elif self.batchnorm_style == "no_sqrt":
                        mvec = gamma * (mvec - mu) / (var + epsilon) + beta

                output_vec = self.activations[m].apply(mvec)
                if search_accuracy and ort_output[search_accuracy_layer] is None and not search_accuracy_depthwise:
                    output_vec_cpu = self.activations[m].apply(mvec_cpu)


                if 'x_scale' in self.auxLayerParams[m] :
                    scale = self.auxLayerParams[m]['x_scale']*self.auxLayerParams[m]['w_scale']/self.auxLayerParams[m]['y_scale']
                    zero_point = self.auxLayerParams[m]['y_zero_point']
                    if len(output_vec.shape) > 1:
                        if isinstance(scale, np.ndarray) and len(scale.shape) > 0 :
                            scale= scale.reshape(-1,1,1)
                        if isinstance(zero_point, np.ndarray) and len(zero_point.shape) > 0 :
                            zero_point = zero_point.reshape(-1,1,1)

                    output_vec =  np.round(output_vec* scale) + zero_point # .astype(np.uint8)
                    if search_accuracy and ort_output[search_accuracy_layer] is None and not search_accuracy_depthwise:
                        output_vec_cpu = np.round(output_vec_cpu* scale) + zero_point
                        ort_output[search_accuracy_layer]=output_vec_cpu
                else:
                    if search_accuracy and ort_output[search_accuracy_layer] is None and not search_accuracy_depthwise:
                        ort_output[search_accuracy_layer]=output_vec_cpu
                
                # if m == 5:
                #         print(output_vec)
                #         raise ValueError("stop")
                
                if branch:
                    output_vecs[m] = output_vec
                    if search_accuracy and not search_accuracy_depthwise:
                        output_vecs[m] = ort_output[search_accuracy_layer]
                

                # compare with onnx
                if search_accuracy and search_accuracy_depthwise == False:
                    # if ort_output == None:
                    #     # if self.auxLayerParams[m]['activation'] is not None:
                    #     #     num_onnx = select_model_inputs_outputs(onnx_model, self.auxLayerParams[m]['activation']['output_name'])
                    #     # else:
                    #     #     num_onnx = select_model_inputs_outputs(onnx_model, self.auxLayerParams[m]['output_name'])
                    #     # save_onnx_model(num_onnx, "pipeline_titanic_numerical.onnx")
                    #     sess_options = ort.SessionOptions()
                    #     sess_options.log_severity_level = 3
                    #     ort_session = ort.InferenceSession(f"{config.model_name}_add_output.onnx", sess_options)
                    #     reshaped_data = np.expand_dims(ort_input, axis=0)
                    #     ort_inputs = {ort_session.get_inputs()[0].name: reshaped_data}
                    #     intermediate_output = ort_session.run(config.output_node_name, ort_inputs)
                       
                    #     ort_output=[]
                    #     for idx, output in enumerate(intermediate_output):
                    #         ort_output.append(np.squeeze(np.array(output)))
                           
                        
                    #     # print(ort_output[search_accuracy_layer].shape)
                    #     if metric == "mse":
                    #         if output_vec.shape != ort_output[search_accuracy_layer].shape:
                    #             raise ValueError("Mismatch in output shape")
                                
                    #         score = ((ort_output[search_accuracy_layer] - output_vec) ** 2).mean()

                    #     elif metric == "cosine":
                    #         array_img1 = np.array(ort_output[search_accuracy_layer])
                    #         array_img2 = np.array(output_vec)
                    #         score = -1*self.cosine_similarity(array_img1, array_img2)
                    #     elif metric == "ssim":
                    #         array_img1 = np.array(ort_output[search_accuracy_layer])
                    #         array_img2 = np.array(output_vec)
                    #         data_range=ort_output[search_accuracy_layer].max() - ort_output[search_accuracy_layer].min()
                    #         score = -1*compare_ssim(array_img1, array_img2, multichannel=True,data_range=data_range)
                    #     elif metric == "cka":
                    #         #reshape 3d to 2d
                    #         if len(ort_output[search_accuracy_layer].shape) == 3:
                    #             # print(ort_output[search_accuracy_layer].shape, output_vec.shape)
                    #             # array_img1 = np.array(ort_output[search_accuracy_layer]).flatten().reshape(-1,1)
                    #             # array_img2 = np.array(output_vec).flatten().reshape(-1,1)
                    #             array_img1 = np.array(ort_output[search_accuracy_layer]).reshape(ort_output[search_accuracy_layer].shape[0],-1)
                    #             array_img2 = np.array(output_vec).reshape(output_vec.shape[0],-1)
                    #             # 이렇게 안하면 이상하게 연산됨
                    #             np.savetxt("cka_ort_output.txt", array_img1)
                    #             np.savetxt("cka_output_vec.txt", array_img2)
                    #             array_img1 = np.loadtxt("cka_ort_output.txt")
                    #             array_img2 = np.loadtxt("cka_output_vec.txt")
                    #             # print(array_img1.shape, array_img2.shape)
                    #             # print(self.linear_CKA(array_img1,array_img2))
                    #             # print(self.linear_CKA(tmp1,tmp2))
                    #             # quit()
                                
                    #         elif len(ort_output[search_accuracy_layer].shape) == 1:
                    #             array_img1 = np.array(ort_output[search_accuracy_layer]).reshape(-1,1)
                    #             array_img2 = np.array(output_vec).reshape(-1,1)
                    #             np.savetxt("cka_ort_output.txt", array_img1)
                    #             np.savetxt("cka_output_vec.txt", array_img2)
                    #             array_img1 = np.loadtxt("cka_ort_output.txt").reshape(-1,1)
                    #             array_img2 = np.loadtxt("cka_output_vec.txt").reshape(-1,1)

                                
                    #         else:
                    #             array_img1 = np.array(ort_output[search_accuracy_layer])
                    #             array_img2 = np.array(output_vec)
                    #         hamming = self.hamming_distance(self.binary_transform(array_img1), self.binary_transform(array_img2))
                    #         score = self.linear_CKA(array_img1,array_img2) * math.log(array_img1.size-hamming)
                    #         # print("score", score)
                    #     elif metric == "hamming":
                    #         mse = ((ort_output[search_accuracy_layer] - output_vec) ** 2).mean()
                    #         array_img1 = np.array(ort_output[search_accuracy_layer])
                    #         array_img2 = np.array(output_vec)
                    #         hamming = self.hamming_distance(self.binary_transform(array_img1), self.binary_transform(array_img2))
                    #         score = hamming*mse
                    #     # cosine_similarity = cosine_similarity(ort_output[search_accuracy_layer], output_vec)
                    #     # data_range=ort_output[search_accuracy_layer].max() - ort_output[search_accuracy_layer].min()

                    #     # array_img1 = np.array(ort_output[search_accuracy_layer])
                    #     # array_img2 = np.array(output_vec)
                    #     # ssim_index = compare_ssim(ort_output[search_accuracy_layer], output_vec, full=False, data_range=data_range)
                    #     # ssim_index = compare_ssim(array_img1, array_img2, multichannel=True,data_range=data_range)
                    #     # cosine = cosine_similarity_3d_scipy(ort_output[search_accuracy_layer], output_vec)
                    #     # cka = self.centered_kernel_alignment(array_img1,array_img2)
                    #     # cosine_similarity = self.cosine_similarity(array_img1, array_img2)
                    #     # haming = self.hamming_distance(self.binary_transform(array_img1), self.binary_transform(array_img2))
                    #     # score = mse
                    #     # print(score)
                    #     # ssim_index, diff = compare_ssim(ort_output[search_accuracy_layer] , output_vec, full=True)
                    #     # print(ssim_index)
                    # else:
                    if metric == "mse":
                        if output_vec.shape != ort_output[search_accuracy_layer].shape:
                            print(output_vec.shape, ort_output[search_accuracy_layer].shape)
                            raise ValueError("Mismatch in output shape")
                        score = ((ort_output[search_accuracy_layer] - output_vec) ** 2).mean()
                    elif metric == "cosine":
                        # array_img1 = np.array(ort_output[search_accuracy_layer])
                        # array_img2 = np.array(output_vec)
                        score = -1*self.cosine_similarity(ort_output[search_accuracy_layer], output_vec)
                    elif metric == "ssim":
                        array_img1 = np.array(ort_output[search_accuracy_layer])
                        array_img2 = np.array(output_vec)
                        data_range=ort_output[search_accuracy_layer].max() - ort_output[search_accuracy_layer].min()
                        score = -1*compare_ssim(array_img1, array_img2, multichannel=True,data_range=data_range)
                    elif metric == "cka":
                        # if shape is 3d, reshape 3d to 2d
                        if len(ort_output[search_accuracy_layer].shape) == 3:
                            # array_img1 = np.array(ort_output[search_accuracy_layer]).flatten().reshape(-1,1)
                            # array_img2 = np.array(output_vec).flatten().reshape(-1,1)
                            # print(ort_output[search_accuracy_layer].shape, output_vec.shape)
                            array_img1 = np.array(ort_output[search_accuracy_layer]).reshape(ort_output[search_accuracy_layer].shape[0],-1)
                            array_img2 = np.array(output_vec).reshape(output_vec.shape[0],-1)
                            # np.savetxt("cka_ort_output.txt", array_img1)
                            # np.savetxt("cka_output_vec.txt", array_img2)
                            # array_img1 = np.loadtxt("cka_ort_output.txt")
                            # array_img2 = np.loadtxt("cka_output_vec.txt")
                            # array_img1 = np.mean(np.array(ort_output[search_accuracy_layer]),axis=(1,2))
                            # array_img2 = np.mean(np.array(output_vec),axis=(1,2))
                        elif len(ort_output[search_accuracy_layer].shape) == 1:
                            array_img1 = np.array(ort_output[search_accuracy_layer]).reshape(-1,1)
                            array_img2 = np.array(output_vec).reshape(-1,1)
                            # np.savetxt("cka_ort_output.txt", array_img1)
                            # np.savetxt("cka_output_vec.txt", array_img2)
                            # array_img1 = np.loadtxt("cka_ort_output.txt").reshape(-1,1)
                            # array_img2 = np.loadtxt("cka_output_vec.txt").reshape(-1,1)
                        else:
                            array_img1 = np.array(ort_output[search_accuracy_layer])
                            array_img2 = np.array(output_vec)
                        # hamming = self.hamming_distance(self.binary_transform(array_img1), self.binary_transform(array_img2))
                        score = self.linear_CKA(array_img1,array_img2) #* math.log(array_img1.size-hamming)
                        # print("score", score)
                    elif metric == "hamming":
                        mse = ((ort_output[search_accuracy_layer] - output_vec) ** 2).mean()
                        array_img1 = np.array(ort_output[search_accuracy_layer])
                        array_img2 = np.array(output_vec)
                        hamming = self.hamming_distance(self.binary_transform(array_img1), self.binary_transform(array_img2))
                        score = hamming*mse

                        # print(ort_output[search_accuracy_layer].shape)
                        # mse = ((ort_output[search_accuracy_layer] - output_vec) ** 2).mean()
                        # cosine_similarity = cosine_similarity(ort_output[search_accuracy_layer], output_vec)

                        # data_range=ort_output[search_accuracy_layer].max() - ort_output[search_accuracy_layer].min()
                        # array_img1 = np.array(ort_output[search_accuracy_layer])
                        # array_img2 = np.array(output_vec)
                        # ssim_index = compare_ssim(array_img1, array_img2, multichannel=True,data_range=data_range)
                        # cka = self.centered_kernel_alignment(array_img1,array_img2)
                        # cosine_similarity = self.cosine_similarity(array_img1, array_img2)
                        # haming = self.hamming_distance(self.binary_transform(array_img1), self.binary_transform(array_img2))
                        # ssim_index = compare_ssim(ort_output[search_accuracy_layer], output_vec, full=False, data_range=data_range)
                        # score = mse
                        # print(score)
                    # intermediate_output_np = np.squeeze(np.array(intermediate_output[0:1]))
                    # mse = ((intermediate_output_np - output_vec) ** 2).mean()
                    # search_accuracy_layer = search_accuracy_layer + 1
                    # print(score, search_accuracy_layer)

                    output_vec = ort_output[search_accuracy_layer]
                    output_vecs[m] = ort_output[search_accuracy_layer]

                    return score, ort_output.copy(), output_vec.copy(), output_vecs.copy(), output_vecs_add.copy(), m
                
                
            #########################################
            #### Pooling layers
            #########################################
            elif self.layerTypes[m] == "pool":
                MPx = self.auxLayerParams[m]["MPx"]
                MPy = self.auxLayerParams[m]["MPy"]
                px_L = self.auxLayerParams[m]["px_L"]
                px_R = self.auxLayerParams[m]["px_R"]
                py_L = self.auxLayerParams[m]["py_L"]
                py_R = self.auxLayerParams[m]["py_R"]
                stride_MP = self.auxLayerParams[m]["stride_MP"]
                poolType = self.auxLayerParams[m]["poolType"]
                avgPool_round = self.auxLayerParams[m]["round"]
                if branch:
                    m_src = sourceLayers[m][0]
                    ivec = output_vecs[m_src].copy()
                    pvec = apply_pool(
                        ivec,
                        MPx,
                        MPy,
                        stride_MP,
                        poolType,
                        px_L,
                        px_R,
                        py_L,
                        py_R,
                        avgPool_round,
                    )
                else:
                    pvec = apply_pool(
                        ivec,
                        MPx,
                        MPy,
                        stride_MP,
                        poolType,
                        px_L,
                        px_R,
                        py_L,
                        py_R,
                        avgPool_round,
                    )

                # Apply batch normalization (rarely used)
                if self.batch_norm[m]:
                    epsilon = self.batch_norm_epsilons[m]
                    gamma, beta, mu, var = self.batch_norm_params[m]
                    if self.batchnorm_style == "sqrt":
                        pvec = gamma * (pvec - mu) / xp.sqrt(var + epsilon) + beta
                    elif self.batchnorm_style == "no_sqrt":
                        pvec = gamma * (pvec - mu) / (var + epsilon) + beta

                # Flatten if pooling was global
                if pvec.shape[1] == 1 and pvec.shape[2] == 1:
                    pvec = flatten_layer(pvec, self.useGPU)

                # Apply activation to output (rarely used)
                output_vec = self.activations[m].apply(pvec)

                if branch:
                    output_vecs[m] = output_vec

            #########################################
            #### Element-wise addition layers
            #########################################
            elif self.layerTypes[m] == "add":
                Nsources = len(sourceLayers[m])
                if Nsources == 2:
                    m_src0 = sourceLayers[m][0]
                    m_src1 = sourceLayers[m][1]
                    if (
                        self.layerTypes[m_src0] == "add"
                        and self.auxLayerParams[m]["splitBeforeBN"]
                    ):
                        ivec0 = output_vecs_add[m_src0].copy()
                    else:
                        ivec0 = output_vecs[m_src0].copy()
                    if (
                        self.layerTypes[m_src1] == "add"
                        and self.auxLayerParams[m]["splitBeforeBN"]
                    ):
                        ivec1 = output_vecs_add[m_src1].copy()
                    else:
                        ivec1 = output_vecs[m_src1].copy()
                    mvec = ivec0 + ivec1
                else:
                    m_src0 = sourceLayers[m][0]
                    ivec0 = output_vecs[m_src0]
                    mvec = ivec0.copy()
                    for q in range(1, Nsources):
                        m_src = sourceLayers[m][q]
                        mvec += output_vecs[m_src]
                mvec = self.activations[m].apply(mvec)
                output_vecs_add[m] = mvec.copy()


                # Apply batch normalization
                if self.batch_norm[m]:
                    epsilon = self.batch_norm_epsilons[m]
                    gamma, beta, mu, var = self.batch_norm_params[m]
                    if self.batchnorm_style == "sqrt":
                        mvec = gamma * (mvec - mu) / xp.sqrt(var + epsilon) + beta
                    elif self.batchnorm_style == "no_sqrt":
                        mvec = gamma * (mvec - mu) / (var + epsilon) + beta

                # Apply activation
                output_vecs[m] = self.activations[m].apply(mvec)
            #########################################
            #### Element-wise multiplication layers
            #########################################
            elif self.layerTypes[m] == "mul":
                Nsources = len(sourceLayers[m])
                if Nsources == 2:
                    m_src0 = sourceLayers[m][0]
                    m_src1 = sourceLayers[m][1]
                    if (
                        self.layerTypes[m_src0] == "mul"
                        and self.auxLayerParams[m]["splitBeforeBN"]
                    ):
                        ivec0 = output_vecs_add[m_src0].copy()
                    else:
                        ivec0 = output_vecs[m_src0].copy()
                    if (
                        self.layerTypes[m_src1] == "mul"
                        and self.auxLayerParams[m]["splitBeforeBN"]
                    ):
                        ivec1 = output_vecs_add[m_src1].copy()
                    else:
                        ivec1 = output_vecs[m_src1].copy()
                    # print(ivec0.shape)
                    # print(ivec1.shape)
                    mvec = ivec0 * ivec1
                else:
                    m_src0 = sourceLayers[m][0]
                    ivec0 = output_vecs[m_src0]
                    mvec = ivec0.copy()
                    for q in range(1, Nsources):
                        m_src = sourceLayers[m][q]
                        mvec *= output_vecs[m_src]

                # Apply batch normalization
                if self.batch_norm[m]:
                    epsilon = self.batch_norm_epsilons[m]
                    gamma, beta, mu, var = self.batch_norm_params[m]
                    if self.batchnorm_style == "sqrt":
                        mvec = gamma * (mvec - mu) / xp.sqrt(var + epsilon) + beta
                    elif self.batchnorm_style == "no_sqrt":
                        mvec = gamma * (mvec - mu) / (var + epsilon) + beta

                # Apply activation
    
                output_vecs[m] = mvec.copy()
               
            #########################################
            #### Concatenation layers
            #########################################
            elif self.layerTypes[m] == "concat":
                Nsources = len(sourceLayers[m])
                m_src0 = sourceLayers[m][0]
                ivec0 = output_vecs[m_src0].copy()
                ovec = ivec0.copy()
                for q in range(1, Nsources):
                    m_src = sourceLayers[m][q]
                    ivec_q = output_vecs[m_src].copy()
                    if (
                        ivec0.shape[1] != ivec_q.shape[1]
                        or ivec0.shape[2] != ivec_q.shape[2]
                    ):
                        raise ValueError("Concat shapes incompatible")
                    ovec = xp.concatenate((ovec, ivec_q), axis=0)
                output_vecs[m] = ovec

            #########################################
            #### Quantization layer (for Nvidia INT4)
            #########################################
            # elif self.layerTypes[m] == "quantize":
            #     shift_bits = self.auxLayerParams[m]["shift_bits"]
            #     output_bits = self.auxLayerParams[m]["output_bits"]
            #     signed = self.auxLayerParams[m]["signed"]
            #     if branch:
            #         m_src = sourceLayers[m][0]
            #         ivec = output_vecs[m_src].copy()
            #         output_vecs[m] = apply_quantization(
            #             ivec,
            #             self.quantization_values[m],
            #             shift_bits,
            #             output_bits,
            #             signed,
            #         )
            #     else:
            #         output_vec = apply_quantization(
            #             ivec,
            #             self.quantization_values[m],
            #             shift_bits,
            #             output_bits,
            #             signed,
            #         )
            elif self.layerTypes[m] == "quantize":
                if 'needless' not in self.auxLayerParams[m]:
                    m_src = sourceLayers[m][0]
                    if m_src == -1 :
                        ivec = indata[one].copy()
                    else :
                        ivec = output_vecs[m_src].copy()
                    scale = self.auxLayerParams[m]["scale"]
                    zero_point = self.auxLayerParams[m]["zero_point"]
                    output_vecs[m] = apply_quantization_onnx(
                            ivec,
                            scale,
                            zero_point
                    )
                    # output_vecs[m] = ivec
                else:
                    m_src = sourceLayers[m][0]
                    ivec = output_vecs[m_src].copy()
                    output_vecs[m] = ivec

      
            elif self.layerTypes[m] == "dequantize":
                if 'needless' not in self.auxLayerParams[m]:
                    m_src = sourceLayers[m][0]
                    ivec = output_vecs[m_src].copy()
                    scale = self.auxLayerParams[m]["scale"]
                    zero_point = self.auxLayerParams[m]["zero_point"]
                    output_vecs[m] = apply_dequantization_onnx(
                            ivec,
                            scale,
                            zero_point
                    )
                    # output_vecs[m] = ivec
                else:
                    m_src = sourceLayers[m][0]
                    if m_src > -1 :
                        ivec = output_vecs[m_src].copy()
                        output_vecs[m] = ivec
             
            elif self.layerTypes[m] == "cast":
                if 'needless' not in self.auxLayerParams[m]:
                    m_src = sourceLayers[m][0]
                    ivec = output_vecs[m_src].copy()
                    cast_type = self.auxLayerParams[m]["casttype"]
                    output_vecs[m] = ivec
                    # if cast_type == 'UINT8':
                        # output_vecs[m] = ivec.astype(np.uint8)
                    # elif cast_type == 'INT32':
                        # output_vecs[m] = ivec.astype(np.int32)


                # else:
                #     m_src = sourceLayers[m][0]
                #     if m_src > -1 :
                #         print(m_src)
                #         ivec = output_vecs[m_src].copy()
                #         output_vecs[m] = ivec
            #########################################
            #### Element-wise scaling layer (for Nvidia INT4)
            #########################################
            elif self.layerTypes[m] == "scale":
                if branch:
                    m_src = sourceLayers[m][0]
                    ivec = output_vecs[m_src].copy()
                if m != (nlayer - 1):
                    output_vec = ivec / self.scale_values[m]
                else:
                    output_vec = self.activations[m].apply(ivec / self.scale_values[m])
                if branch:
                    output_vecs[m] = output_vec

            #########################################
            #### Flatten layer
            #########################################
            elif self.layerTypes[m] == "flatten":
                m_src = sourceLayers[m][0]
                ivec = output_vecs[m_src].copy()
                output_vecs[m] = flatten_layer(ivec, self.useGPU)
            
            elif self.layerTypes[m] == "reducemean":
                m_src = sourceLayers[m][0]
                ivec = output_vecs[m_src].copy()
                output_vecs[m] = reducemean_layer(ivec, self.auxLayerParams[m]["axes"], self.auxLayerParams[m]["keepdims"],self.useGPU)
            #########################################
            #### Flatten layer (for network input)
            #########################################
            elif self.layerTypes[m] == "flatten_input":
                ivec = indata[one].copy()
                output_vecs[m] = flatten_layer(ivec, self.useGPU)

            #########################################
            #### space2depth layer (TensorFlow/Pytorch)
            #########################################
            # This layer type is not currently being supported in keras_parser()
            elif self.layerTypes[m] == "space2depth":
                if branch:
                    m_src = sourceLayers[m][0]
                    ivec = output_vecs[m_src].copy()
                    output_vecs[m] = space_to_depth(ivec, 2)
                else:
                    output_vec = space_to_depth(ivec, 2)

            # Clear unneeded activations from memory
            if branch and memory_window > 0 and m >= memory_window:
                output_vecs[m - memory_window] = None
                output_vecs_add[m - memory_window] = None
        # Network output
        if branch:
            network_output = output_vecs[-1]
        else:
            network_output = output_vec

        if len(network_output.shape) > 1:
            network_output = flatten_layer(network_output, self.useGPU)
        if self.whetstone:
            network_output = np.asarray(
                decode_from_key(self.decoder_mat, network_output, self.useGPU),
            )

        # Answer
        actual = int(answers[one])

        # Top-k accuracy:
        # If topk is an integer, returns a single value for top-k accuracy
        # If topk is a list, tuple, or array, returns a list of accuracies for different k
        if len(network_output) > 1:
            if type(topk) is int:
                if topk == 1:
                    index = network_output.argmax()
                    result = 1 if index == actual else 0
                else:
                    indices = np.argpartition(network_output, -topk)[-topk:]
                    result = 1 if actual in indices else 0
            else:
                result = np.zeros(len(topk))
                for j in range(len(topk)):
                    indices = np.argpartition(network_output, -topk[j])[-topk[j] :]
                    result[j] = 1 if actual in indices else 0
        else:
            # Single output neuron case (for binary classification tasks)
            output = 0 if network_output < 0.5 else 1
            result = 1 if output == actual else 0

        if return_network_output:
            return result, network_output
        else:
            return result, None

    # -------------------------------------------------------
    # Save the profiled DAC and ADC inputs for all layers
    # -------------------------------------------------------
    def save_profiled_data(self, profiling_folder, profile_in_dnn):
        # Save the profiled activations to the destination folder
        if self.profile_DAC_inputs:
            print("Saving DAC input data")
            for ilayer in range(self.nlayer):
                if self.layerTypes[ilayer] not in ("conv", "dense"):
                    continue
                np.save(
                    profiling_folder + "dac_inputs_layer" + str(ilayer) + ".npy",
                    self.profiled_values[ilayer],
                )

        # Save ADC inputs
        if self.profile_ADC_inputs:
            print("Saving ADC input data")
            for ilayer in range(self.nlayer):
                if self.layerTypes[ilayer] not in ("conv", "dense"):
                    continue

                # Save profiled ADC input values stored inside cores
                #   Cores: same file
                #   Input bits: same file
                #   Weight slices: separate files
                if self.ncores[ilayer].params.simulation.analytics.profile_adc_inputs:
                    cores = self.ncores[ilayer].cores
                    Ncores_r = self.ncores[ilayer].num_cores_row
                    Ncores_c = self.ncores[ilayer].num_cores_col

                    # Profiling ADC inputs with no weight bit slicing
                    # Dimensions: (# cores row) x (# cores col) x (# input bit slices) x (# outputs)
                    if self.ncores[ilayer].params.core.style != CoreStyle.BITSLICED:
                        core0_outputs = cores[0][0].adc_inputs
                        all_core_outputs = xp.zeros(
                            (
                                Ncores_r,
                                Ncores_c,
                                core0_outputs.shape[0],
                                core0_outputs.shape[1],
                            ),
                        )
                        for r in range(Ncores_r):
                            for c in range(Ncores_c):
                                if cores[r][c].adc_inputs.shape != core0_outputs.shape:
                                    input_shape = cores[r][c].adc_inputs.shape
                                    target_shape = all_core_outputs[r, c, :, :].shape
                                    adjusted_input = np.zeros(target_shape)
                                    rows_to_copy = min(input_shape[0], target_shape[0])
                                    cols_to_copy = min(input_shape[1], target_shape[1])
                                    adjusted_input[:rows_to_copy, :cols_to_copy] = cores[r][c].adc_inputs[:rows_to_copy, :cols_to_copy]
                                    all_core_outputs[r, c, :, :] = adjusted_input
                                else:
                                    all_core_outputs[r, c, :, :] = cores[r][c].adc_inputs

                        np.save(
                            profiling_folder
                            + "adc_inputs_layer"
                            + str(ilayer)
                            + ".npy",
                            all_core_outputs,
                        )

                    # For weight bit slicing, ADC inputs are resolved by weight bit slice in different files
                    # For each file, dimensions: (# cores row) x (# cores col) x (# input bit slices) x (# outputs)
                    else:
                        for k in range(
                            self.ncores[ilayer].params.core.bit_sliced.num_slices,
                        ):
                            core0_outputs_k = cores[0][0].adc_inputs[k, :, :]
                            bitslice_outputs_k = xp.zeros(
                                (
                                    Ncores_r,
                                    Ncores_c,
                                    core0_outputs_k.shape[0],
                                    core0_outputs_k.shape[1],
                                ),
                            )
                            for r in range(Ncores_r):
                                for c in range(Ncores_c):
                                    if cores[r][c].adc_inputs[k, :, :].shape != bitslice_outputs_k[r, c, :, :].shape:
                                        input_shape = cores[r][c].adc_inputs[k, :, :].shape
                                        target_shape = bitslice_outputs_k[r, c, :, :].shape
                                        adjusted_input = np.zeros(target_shape)
                                        rows_to_copy = min(input_shape[0], target_shape[0])
                                        cols_to_copy = min(input_shape[1], target_shape[1])
                                        adjusted_input[:rows_to_copy, :cols_to_copy] = cores[r][c].adc_inputs[k, :rows_to_copy, :cols_to_copy]
                                        bitslice_outputs_k[r, c, :, :] = adjusted_input
                                    else:
                                        bitslice_outputs_k[r, c, :, :] = cores[r][c].adc_inputs[k, :, :]
                            np.save(
                                profiling_folder
                                + "adc_inputs_layer"
                                + str(ilayer)
                                + "_slice"
                                + str(k)
                                + ".npy",
                                bitslice_outputs_k,
                            )

                # Save profiled DAC/ADC inputs that are stored in DNN object
                else:
                    if not profile_in_dnn:
                        # Should not be accessible
                        raise ValueError(
                            "ADC inputs profiled neither in core nor on DNN",
                        )
                    # Handle ReLU-aware profiling outputs
                    output_values = self.profiled_values[ilayer]
                    # Convert to crossbar ADC units
                    output_values /= self.ncores[ilayer].cores[0][0].mvm_out_scale
                    # Add axis for the single core and single input bit slice
                    output_values = output_values[None, None, None, :]
                    np.save(
                        profiling_folder + "adc_inputs_layer" + str(ilayer) + ".npy",
                        output_values,
                    )

    # -------------------------------------------------------
    # Import bias weights if not using analog core to add bias
    # -------------------------------------------------------
    def import_digital_bias(self, weight_dict, bias_bits):
        for ilayer in range(self.nlayer):
            if self.layerTypes[ilayer] not in ("conv", "dense"):
                continue

            # Extract raw tensors from Keras model
            Wi_0 = weight_dict[self.auxLayerParams[ilayer]["name"]]

            # Extract the bias weights
            if self.digital_bias[ilayer]:
                Wbias = Wi_0[1].astype(xp.float32)

                # Quantize the bias values (set bias_bits = 0 to disable)
                # Range is set by the minimum and maximum
                # Options: for bias_bit,
                #   0       : no quantization
                #   int > 0 : quantize to this number of bits with range set by min and max bias values
                #   "adc"   : set the range to the ADC range and the bias_bits to the adc resolution
                #       If a bigger range is needed than ADC, set the range to a power of 2 times the ADC range
                #       and add the appropriate number of bits
                if bias_bits != "adc":
                    if bias_bits > 0:
                        Bmax = np.max(Wbias)
                        Bmin = np.min(Wbias)
                        qmult = (2**bias_bits) / (Bmax - Bmin)
                        Wbias -= Bmin
                        Wbias *= qmult
                        Wbias = np.rint(Wbias, out=Wbias)
                        Wbias /= qmult
                        Wbias += Bmin

                else:
                    # Find the range of bias weights
                    Bmax = np.max(Wbias)
                    Bmin = np.min(Wbias)

                    # Find ADC range of corresponding layer
                    adc_bits = self.ncores[ilayer].params.xbar.adc.mvm.bits
                    adc_min = (
                        self.ncores[ilayer].cores[0][0].adc.mvm.min
                        * self.ncores[ilayer].cores[0][0].mvm_out_scale
                    )
                    adc_max = (
                        self.ncores[ilayer].cores[0][0].adc.mvm.max
                        * self.ncores[ilayer].cores[0][0].mvm_out_scale
                    )

                    if adc_bits > 0:
                        # If the layer has multiple cores, its range effectively expands when added
                        if self.ncores[ilayer].Ncores > 1:
                            expand_bits = np.ceil(np.log2(self.ncores[ilayer].Ncores))
                            adc_min *= pow(2, expand_bits)
                            adc_max *= pow(2, expand_bits)
                            adc_bits += expand_bits

                        if Bmax < adc_max and Bmin > adc_min:
                            # Bias range is contained inside ADC range
                            b_min = adc_min
                            b_max = adc_max
                            nbits = adc_bits

                        elif Bmax > adc_max and Bmin > adc_min:
                            # Bias max is larger than ADC max
                            extend_bits = np.ceil(
                                np.log2((Bmax - adc_min) / (adc_max - adc_min)),
                            )
                            b_max = adc_min + (adc_max - adc_min) * pow(2, extend_bits)
                            b_min = adc_min
                            nbits = adc_bits + extend_bits

                        elif Bmax < adc_max and Bmin < adc_min:
                            # Bias min is smaller than ADC min
                            extend_bits = np.ceil(
                                np.log2((adc_max - Bmin) / (adc_max - adc_min)),
                            )
                            b_max = adc_max
                            b_min = adc_max - (adc_max - adc_min) * pow(2, extend_bits)
                            nbits = adc_bits + extend_bits

                        elif Bmax > adc_max and Bmin < adc_min:
                            # Bias limits are beyond both limits of ADC
                            # First extend min, then extend max
                            extend_bits_min = np.ceil(
                                np.log2((adc_max - Bmin) / (adc_max - adc_min)),
                            )
                            b_min = adc_max - (adc_max - adc_min) * pow(
                                2, extend_bits_min,
                            )
                            extend_bits_max = np.ceil(
                                np.log2((Bmax - b_min) / (adc_max - b_min)),
                            )
                            b_max = b_min + (adc_max - b_min) * pow(2, extend_bits_max)
                            nbits = adc_bits + extend_bits_min + extend_bits_max

                        qmult = (2**nbits) / (b_max - b_min)
                        Wbias -= b_min
                        Wbias *= qmult
                        Wbias = np.rint(Wbias, out=Wbias)
                        Wbias /= qmult
                        Wbias += b_min

                if self.layerTypes[ilayer] == "conv":
                    Wbias = Wbias.reshape((len(Wbias), 1, 1))
                Wbias = xp.array(Wbias)
                self.bias_values[ilayer] = Wbias

    # -------------------------------------------------------
    # Profiling of DAC and ADC inputs
    # -------------------------------------------------------
    def update_adc_dac_statistics(self, m, ivec, mvec):
        # Collect all ADC input values (beware of memory consumption!)
        # This function is used when input or weight bit slices are not enabled
        # If layer has no activation, do not account for a ReLU during profiling
        hasRelu = self.activations[m].style == RECTLINEAR
        singleCore = self.ncores[m].Ncores == 1

        if self.profile_DAC_inputs:
            self.profiled_values[m] = xp.concatenate(
                (self.profiled_values[m], xp.array(ivec.flatten(), dtype=xp.float32)),
            )

        elif self.profile_ADC_reluAware:
            # If this is false, use profiled ADC inputs in the core
            if hasRelu and singleCore:
                ADC_inputs = mvec[mvec > -self.bias_values[m]].flatten()
                ADC_inputs = xp.array(ADC_inputs, dtype=xp.float32)
                self.profiled_values[m] = xp.concatenate(
                    (self.profiled_values[m], ADC_inputs),
                )

        # This should not be accessible
        elif self.profile_ADC_inputs and not self.profile_ADC_reluAware:
            raise ValueError(
                "Attemped to profile ADC inputs inside DNN without ReLU-aware setting",
            )

    # -------------------------------------------------------
    # Used for Nvidia INT4 model only
    # -------------------------------------------------------
    def import_quantization(self, weight_dict):
        for ilayer in range(self.nlayer):
            if self.layerTypes[ilayer] != "quantize":
                continue
            W_q = weight_dict[self.auxLayerParams[ilayer]["name"]][0]
            W_q = W_q.reshape((len(W_q), 1, 1))
            W_q = xp.array(W_q)
            self.quantization_values[ilayer] = W_q

    def import_scale(self, weight_dict):
        for ilayer in range(self.nlayer):
            if self.layerTypes[ilayer] != "scale":
                continue
            W_s = weight_dict[self.auxLayerParams[ilayer]["name"]][0]
            W_s = xp.array(W_s)
            self.scale_values[ilayer] = W_s
