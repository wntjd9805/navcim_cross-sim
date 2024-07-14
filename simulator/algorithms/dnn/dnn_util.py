#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#
import numpy as np
from scipy.special import softmax
from ...backend import ComputeBackend
import torch
import torch.nn.functional as F

xp = ComputeBackend()


def init_GPU_util(useGPU):
    global as_strided
    if useGPU:
        from cupy.lib.stride_tricks import as_strided
    else:
        from numpy.lib.stride_tricks import as_strided


# Function for applying max and average pooling
def apply_pool(
    matrix, MPx, MPy, stride_MP, poolType, px_L, px_R, py_L, py_R, avgPool_round,
):
    """Perform a max pool operation on a matrix, taking the max over MPx x MPy block with a given stride
    Assume the stride is equal to the kernel size
    This function is not tied to the convolution core object.
    """
    # No maxpool case

    # if px_L > 0 or px_R > 0 or py_L > 0 or py_R > 0:
    #     matrix = np.pad(matrix, ((0, 0), (px_L, px_R), (py_L, py_R)), "constant")

    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)
    
    # Add batch and channel dimensions if they don't exist
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0).unsqueeze(0)
    elif matrix.dim() == 3:
        matrix = matrix.unsqueeze(0)

    # Apply the appropriate pooling function

    
    if poolType == 'max':
        pooled = F.max_pool2d(matrix, kernel_size=(MPx, MPy), stride=stride_MP, padding=(px_L, py_L))
    elif poolType == 'avg':
        pooled = F.avg_pool2d(matrix, kernel_size=(MPx, MPy), stride=stride_MP, padding=(px_L, py_L))
        if avgPool_round:
            pooled = torch.floor(pooled)

    if pooled.size(0) == 1:
        pooled = pooled.squeeze(0)
    pooled_np = pooled.numpy()
    return pooled_np
    # Padding
    # if px_L > 0 or px_R > 0 or py_L > 0 or py_R > 0:
    #     matrix = xp.pad(matrix, ((0, 0), (px_L, px_R), (py_L, py_R)), "constant")

    # if MPx == 1 and MPy == 1 and stride_MP == 1:
    #     return matrix

    # Nc, Nx, Ny = matrix.shape

    # # If matrix size is not divisible by MaxPool stride, cut off bottom and/or right edges of matrix
    # # This is the TensorFlow/Keras convention
    # if Nx % stride_MP != 0 or Ny % stride_MP != 0:
    #     x_extra = Nx % stride_MP
    #     y_extra = Ny % stride_MP
    #     matrix = matrix[:, : (Nx - x_extra), : (Ny - y_extra)]

    # if MPx == stride_MP and MPy == stride_MP:
    #     # This is slightly faster for non-overlapping pooling (the common case)
    #     Bx = Nx // MPx
    #     By = Ny // MPy
    #     if poolType == "max":
    #         return matrix.reshape(Nc, Bx, MPx, By, MPy).max(axis=(2, 4))
    #     elif poolType == "avg":
    #         if not avgPool_round:
    #             return matrix.reshape(Nc, Bx, MPx, By, MPy).mean(axis=(2, 4))
    #         else:
    #             return xp.floor(matrix.reshape(Nc, Bx, MPx, By, MPy).mean(axis=(2, 4)))

    # else:
    #     # Re-written for channel first order
    #     Mout_shape = (Nc, (Nx - MPx) // stride_MP + 1, (Ny - MPy) // stride_MP + 1)
    #     kernel_size = (1, MPx, MPy)
    #     M0 = as_strided(
    #         matrix,
    #         shape=Mout_shape + kernel_size,
    #         strides=(
    #             matrix.strides[0],
    #             stride_MP * matrix.strides[1],
    #             stride_MP * matrix.strides[2],
    #         )
    #         + matrix.strides,
    #     )
    #     M0 = M0.reshape(-1, *kernel_size)
    #     if poolType == "max":
    #         return M0.max(axis=(1, 2)).reshape(Mout_shape)
    #     elif poolType == "avg":
    #         if not avgPool_round:
    #             return M0.mean(axis=(1, 2)).reshape(Mout_shape)
    #         else:
    #             return np.floor(M0.mean(axis=(1, 2)).reshape(Mout_shape))


def flatten_layer(matrix, useGPU):
    """Flatten a 3D matrix (Nx,Ny,Nchannels) to a 1D vector, in a way that is identical with a Flatten layer in Keras."""
    if len(matrix.shape) == 1:
        return matrix
    #juseong
    # matrix = xp.transpose(matrix, (1, 2, 0))
    if useGPU:
        return matrix.flatten()
    else:
        return matrix.flatten()

def reducemean_layer(matrix,axis,keepdims ,useGPU):
    """Flatten a 3D matrix (Nx,Ny,Nchannels) to a 1D vector, in a way that is identical with a Flatten layer in Keras."""
    
    mean_matrix = matrix.mean(axis=axis, keepdims=keepdims)
    
    # Flatten the result if keepdims is False, otherwise return as is
    if not keepdims:
        return mean_matrix.flatten()
    else:
        return mean_matrix
    
# Applies a space2depth operation, assuming NCHW data format
# See: https://www.tensorflow.org/api_docs/python/tf/nn/space_to_depth
def space_to_depth(x, block_size):
    x = x.transpose((1, 2, 0))
    height, width, depth = x.shape
    reduced_height = height // block_size
    reduced_width = width // block_size
    y = x.reshape(reduced_height, block_size, reduced_width, block_size, depth)
    z = xp.swapaxes(y, 1, 2).reshape(reduced_height, reduced_width, -1)
    z = z.transpose((2, 0, 1))
    return z


# Nvidia custom quantization/de-quantization layer
# Applies fake quantization
# See: https://github.com/mlperf/inference_results_v0.5/tree/master/open/NVIDIA
def apply_quantization(x, W, shift_bits, output_bits, signed):
    ymax = pow(2, output_bits) - 1
    if not signed:
        ymin = 0
        y = x * W + pow(2, shift_bits) / 2
        y /= pow(2, shift_bits)
        y = xp.floor(y)

    else:
        ymin = -pow(2, output_bits)
        y_pos = x * W + pow(2, shift_bits) / 2
        y_pos /= pow(2, shift_bits)
        y_pos = xp.floor(y_pos)
        y_neg = x * W - pow(2, shift_bits) / 2
        y_neg /= pow(2, shift_bits)
        y_neg = xp.ceil(y_neg)
        y = y_pos * (x >= 0) + y_neg * (x < 0)

    y.clip(ymin, ymax, out=y)

    return y

def apply_quantization_onnx(x, scale, zero_point):
    y = np.round(x / scale).astype(np.int32) + zero_point
    return y
def apply_dequantization_onnx(x, scale, zero_point):
    y = ( x - zero_point ) * scale 
    return y

# Output function for Whetstone models


def decode_from_key(key, input_vec, useGPU):
    """Decodes a vector using the specified key.

    # Arguments
        key: Key used for decoding (ndarray)
        input_vec: Vector of size key.shape[1] to be decoded.

    # Returns
        Decoded one-hot vector.
    """
    # return [1*(np.argmax(np.matmul(2*key-1,2*input_vec-1))==i) for i in range(0, key.shape[0])]
    if useGPU:
        x = cp.asnumpy(input_vec)
    return softmax(np.dot(2 * x - 1, 2 * key - 1))
