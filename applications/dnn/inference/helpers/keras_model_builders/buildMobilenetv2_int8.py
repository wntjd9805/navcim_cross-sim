#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

#
# This file has been adapted from Keras:
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py
#

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as backend
import tensorflow.keras.models as models
import tensorflow.keras.utils as keras_utils
import tensorflow.keras.layers as layers
from tensorflow.keras.applications import imagenet_utils


def preprocess_input(x, **kwargs):
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v





def MobileNetV2_int8(weight_dict,weight_limits_dict,
              include_top=True,
              classes=1000,
              **kwargs):
   
    alpha = 1.0
    depth_multiplier = 1

    # Determine proper input shape and default size.
    default_size = 224

    input_shape = _obtain_input_shape(None,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=None)
    print(input_shape)
    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]


    img_input = layers.Input(shape=input_shape)

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    first_block_filters = _make_divisible(32 * alpha, 8)

    
    x = layers.ZeroPadding2D(padding=correct_pad(backend, img_input, 3),
                             name='Conv1_pad')(img_input)
    x = layers.Conv2D(first_block_filters,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=True,
                      name='Conv1')(x)
    # x = layers.BatchNormalization(axis=channel_axis,
    #                               epsilon=1e-3,
    #                               momentum=0.999,
    #                               name='bn_Conv1')(x)
    x = layers.ReLU(6., name='Conv1_relu')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=9)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=12)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                            expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=15)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                            expansion=6, block_id=16)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = layers.Conv2D(last_block_filters,
                      kernel_size=1,
                      use_bias=True,
                      name='Conv_1')(x)
    # x = layers.BatchNormalization(axis=channel_axis,
    #                               epsilon=1e-3,
    #                               momentum=0.999,
    #                               name='Conv_1_bn')(x)
    x = layers.ReLU(6., name='out_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(classes, activation='softmax',
                         use_bias=True, name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
  
    inputs = img_input

    # Create model.
    model = models.Model(inputs, x,
                         name='mobilenetv2_%0.2f_%s' % (alpha, rows))

    # Load weights.
    for i in range(len(model.layers)):
        layer_name = model.layers[i].get_config()['name']
        if layer_name in weight_dict and ('relu' not in layer_name and 'pad' not in layer_name and 'add' not in layer_name and 'global_average_pooling2d' not in layer_name):
            print(layer_name)
            weights = weight_dict[layer_name]
            if layer_name in weight_limits_dict.keys():
                if layer_name + '_BN' in weight_dict.keys():
                    gamma = weight_dict[layer_name+'_BN']['gamma']
                    beta  = weight_dict[layer_name+'_BN']['beta']
                    mu = weight_dict[layer_name+'_BN']['mu']
                    var = weight_dict[layer_name+'_BN']['var']
                    epsilon = 1e-3
                    Wbias = np.zeros(weights['weights'].shape[-1])

                    if "depthwise" in layer_name:
                        Wm = gamma[None,None,:,None]*weights['weights']/np.sqrt(var[None,None,:,None] + epsilon)
                        Wbias = (gamma[None,None,:,None]/np.sqrt(var[None,None,:,None] + epsilon))*(Wbias-mu[None,None,:,None]) + beta[None,None,:,None]
                        Wbias = np.squeeze(Wbias)
                    else:
                        Wm = gamma*weights['weights']/np.sqrt(var + epsilon)
                        Wbias = (gamma/np.sqrt(var + epsilon))*(Wbias-mu) + beta
                elif 'bn_' + layer_name in weight_dict.keys():
                    gamma = weight_dict['bn_' + layer_name]['gamma']
                    beta  = weight_dict['bn_' + layer_name]['beta']
                    mu = weight_dict['bn_' + layer_name]['mu']
                    var = weight_dict['bn_' + layer_name]['var']
                    epsilon = 1e-3
                    Wbias = np.zeros(weights['weights'].shape[-1])

                    if "depthwise" in layer_name:
                        Wm = gamma[None,None,:,None]*weights['weights']/np.sqrt(var[None,None,:,None] + epsilon)
                        Wbias = (gamma[None,None,:,None]/np.sqrt(var[None,None,:,None] + epsilon))*(Wbias-mu[None,None,:,None]) + beta[None,None,:,None]
                        Wbias = np.squeeze(Wbias)
                    else:
                        Wm = gamma*weights['weights']/np.sqrt(var + epsilon)
                        Wbias = (gamma/np.sqrt(var + epsilon))*(Wbias-mu) + beta

                elif layer_name + '_bn' in weight_dict.keys():
                    gamma = weight_dict[layer_name+'_bn']['gamma']
                    beta  = weight_dict[layer_name+'_bn']['beta']
                    mu = weight_dict[layer_name+'_bn']['mu']
                    var = weight_dict[layer_name+'_bn']['var']
                    epsilon = 1e-3
                    Wbias = np.zeros(weights['weights'].shape[-1])

                    if "depthwise" in layer_name:
                        Wm = gamma[None,None,:,None]*weights['weights']/np.sqrt(var[None,None,:,None] + epsilon)
                        Wbias = (gamma[None,None,:,None]/np.sqrt(var[None,None,:,None] + epsilon))*(Wbias-mu[None,None,:,None]) + beta[None,None,:,None]
                        Wbias = np.squeeze(Wbias)
                    else:
                        Wm = gamma*weights['weights']/np.sqrt(var + epsilon)
                        Wbias = (gamma/np.sqrt(var + epsilon))*(Wbias-mu) + beta
                else:
                    # print(layer_name)
                    # print(weights)
                    Wm = weights['weights']
                    print(Wm.shape)
                    if 'bias' in weights.keys():
                        Wbias = weights['bias']        
                    else: 
                        Wbias = np.zeros(weights['weights'].shape[-1])

                weight_limits = weight_limits_dict[layer_name]

                # Quantize
                # Wm = tf.quantization.fake_quant_with_min_max_vars(Wm,weight_limits['min'],weight_limits['max'],num_bits=8,narrow_range=True)

                weights = [Wm,Wbias]
            # print(weights)
            
            model.layers[i].set_weights(weights)
    # exit()
    return model


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=True,
                          activation=None,
                          name=prefix + 'expand')(x)
        # x = layers.BatchNormalization(axis=channel_axis,
        #                               epsilon=1e-3,
        #                               momentum=0.999,
        #                               name=prefix + 'expand_BN')(x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, 3),
                                 name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               activation=None,
                               use_bias=True,
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    # x = layers.BatchNormalization(axis=channel_axis,
    #                               epsilon=1e-3,
    #                               momentum=0.999,
    #                               name=prefix + 'depthwise_BN')(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=True,
                      activation=None,
                      name=prefix + 'project')(x)
    # x = layers.BatchNormalization(axis=channel_axis,
    #                               epsilon=1e-3,
    #                               momentum=0.999,
    #                               name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[-1]) + ' input channels.')
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting `include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                   (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                   (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape


def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))