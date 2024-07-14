#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np
import os, sys, pickle, argparse
#To import parameters
sys.path.append("../../../simulator/")
#To import simulator
sys.path.append("../../../")
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"

from interface.inference_net import set_params, inference
from interface.keras_parser import get_keras_metadata
from interface.onnx_parser import get_onnx_metadata
from interface.dnn_setup import augment_parameters, build_keras_model, model_specific_parameters, \
    get_xy_parallel, get_xy_parallel_parasitics, load_adc_activation_ranges, check_profiling_settings
from interface.config_message import print_configuration_message

# ==========================
# ==== Load config file ====
# ==========================

import inference_config as config
import onnx
import os

navcim_dir = os.getenv('NAVCIM_DIR')
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--model', default='VGG16', help='VGG16|ResNet50|NasNetA|LFFD')
parser.add_argument('--ntest', type=int, default=50)
parser.add_argument('--ntest_batch', type=int, default=50)
parser.add_argument('--Nslices', type=int, default=1, help='Nslices')
parser.add_argument('--NrowsMax', type=int, default=1152, help='NrowsMax')
parser.add_argument('--NcolsMax', type=int, default=0, help='NcolsMax')
parser.add_argument('--profile_ADC_inputs',type=int, default=0, help='profile_ADC_inputs')
parser.add_argument('--profile_DAC_inputs',type=int ,default=0, help='profile_ADC_inputs')
parser.add_argument('--cell_bits', type=int, default=7, help='cell_bits')
args = parser.parse_args()

config.model_name = args.model
config.ntest = args.ntest 
config.ntest_batch = args.ntest_batch
config.Nslices = args.Nslices
config.NrowsMax = args.NrowsMax
config.NcolsMax = args.NcolsMax
config.Nslices = np.ceil((config.weight_bits-1) / args.cell_bits).astype(int)
# ============================
# ==== Profiling settings ====
# ============================


# Profile ADC inputs
print(args.profile_ADC_inputs)
print(args.profile_DAC_inputs)
if args.profile_ADC_inputs == 1 and args.profile_DAC_inputs == 0:
    profile_ADC_inputs = True
    profile_DAC_inputs = False
    config.adc_bits = 0
    
# Profile activations
elif args.profile_DAC_inputs == 1 and args.profile_ADC_inputs == 0:
    profile_DAC_inputs = True
    profile_ADC_inputs = False
    config.adc_bits = 0
    config.dac_bits = 0
else:
    raise ValueError("Cannot profile both DAC and ADC inputs")
#################

# Discard ADC inputs that would be zero'ed out by ReLU
# This setting is only used for cores with:
# (1) no input bit slicing, (2) no weight bit slicing, (3) no partitioning, (4) uses ReLU
# Only used if profile_ADC_inputs is true
profile_ADC_reluAware = False

# Use the calibration set for profiling
# If True:
#   For non-ImageNet, ntest random images are sampled from the training set (see dataset_loaders)
#   For ImageNet, pre-processed MLPerf images are used (must be available)
# If False (not recommended)
#   The same images will be used for calibration as used for a normal simulation, in the same order
calibration = True

check_profiling_settings(config, profile_DAC_inputs, profile_ADC_inputs, profile_ADC_reluAware)

# Folder to save profiling .npy results: make sure it exists
# NOTE: the descriptors used in the file name template do not fully specify the settings needed to produce the ADC input values
# Many other parameters, for example fold_batchnorm, are also important
if profile_ADC_inputs:
    ibit_msg = ("_ibits" if config.input_bitslicing else "")
    relu_msg = ("_reluAware" if profile_ADC_reluAware else "")
    profiling_folder = "./adc/profiled_adc_inputs/"+\
        str(config.task)+"_"+str(config.model_name)+"_"+str(config.NrowsMax)+"rows_"+str(config.NcolsMax)+"cols_"+str(config.Nslices)+"slices_"+config.style+ibit_msg+relu_msg+"/"

elif profile_DAC_inputs:
    profiling_folder = "./adc/profiled_dac_inputs/"+str(config.task)+"_"+str(config.model_name)+"/"

else:
    profiling_folder = None

if profiling_folder is not None and not os.path.isdir(profiling_folder):
    os.makedirs(profiling_folder)
    print("Created new profiling directory: "+profiling_folder)

# ===================
# ==== GPU setup ====
# ===================

# Restrict tensorflow GPU memory usage
os.environ["CUDA_VISIBLE_DEVICES"]=str(-1)
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for k in range(len(gpu_devices)):
    tf.config.experimental.set_memory_growth(gpu_devices[k], True)

# Set up GPU
if config.useGPU:
    import cupy
    os.environ["CUDA_VISIBLE_DEVICES"]=str(config.gpu_num)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    cupy.cuda.Device(0).use()

# =====================================
# ==== Import neural network model ====
# =====================================

# Build Keras model and prepare CrossSim-compatible topology metadata
# keras_model = build_keras_model(config.model_name,show_model_summary=config.show_model_summary)
# layerParams, sizes = get_keras_metadata(keras_model,task=config.task,debug_graph=False)
quantize = False
if config.model_name== 'VGG11':
    onnx_model = onnx.load(f"{navcim_dir}/cross-sim/applications/dnn/inference/model/vgg11.onnx")
elif config.model_name== 'Resnet50':
    onnx_model = onnx.load(f"{navcim_dir}/cross-sim/applications/dnn/inference/model/resnet50_int8.onnx")
    quantize =  True
elif config.model_name== 'MobileNetV2':
    onnx_model = onnx.load(f"{navcim_dir}/cross-sim/applications/dnn/inference/model/mobilenetv2_int8.onnx")
    quantize =  True
elif config.model_name== 'SqueezeNet':
    onnx_model = onnx.load(f"{navcim_dir}/cross-sim/applications/dnn/inference/model/SqueezeNet.onnx")
elif config.model_name== 'RegNetY':
    onnx_model = onnx.load(f"{navcim_dir}/cross-sim/applications/dnn/inference/model/regnet_y_8gf.onnx")



layerParams, sizes ,weight_for_quantized,output_node_name= get_onnx_metadata(onnx_model,config.model_name, quantize, debug_graph=False, search_accuracy =False)

# Count the total number of layers and number of MVM layers
Nlayers = len(layerParams)
config.Nlayers_mvm = np.sum([(layerParams[j]['type'] in ('conv','dense')) for j in range(Nlayers)])

for i , pa in enumerate(layerParams):
    print(i ,layerParams[i]['name'],layerParams[i]['type'])

# ===================================================================
# ======= Parameter validation and model-specific parameters ========
# ===================================================================
# General parameter check
config = augment_parameters(config)
# Parameters specific to some neural networks
config, positiveInputsOnly = model_specific_parameters(config)

relu_list = []
if config.input_framework =='onnx':
    for j in range(Nlayers) : 
        if layerParams[j]['type'] in ('conv','dense'):
            if layerParams[j]['activation'] is not None:
                # if layerParams[j]['activation']['type'] in ('CLIP','RECTLINEAR','SIGMOID','SOFTMAX'):
                #     positiveInputsOnly[j] = True
                if layerParams[j]['activation']['type'] in ('CLIP','RECTLINEAR'):
                    relu_list.append(1)
                else:
                    relu_list.append(0)
            else:
                relu_list.append(0)
            
# =========================================================
# ==== Load calibrated ranges for ADCs and activations ====
# =========================================================

adc_ranges, dac_ranges,positiveInputsOnly= load_adc_activation_ranges(config,positiveInputsOnly)

# =======================================
# ======= GPU performance tuning ========
# =======================================

# Convolutions: number of sliding windows along x and y to compute in parallel
xy_pars = get_xy_parallel(config, disable=config.disable_SW_packing)

# ================================
# ========= Start sweep ==========
# ================================

if profile_DAC_inputs:
    print("*** Profiling DAC inputs ***")
elif profile_ADC_inputs:
    if not profile_ADC_reluAware:
        print("*** Profiling ADC inputs ***")
    else:
        print("*** Profiling ADC inputs (ReLU-aware) ***")

# Display the chosen simulation settings
print_configuration_message(config)

paramsList, layerParamsCopy = Nlayers*[None], Nlayers*[None]
j_mvm, j_conv = 0, 0 # counter for MVM and conv layers

# ===================================================
# ==== Compute and set layer-specific parameters ====
# ===================================================
nrow =[]
for j in range(Nlayers):

    # For a layer that must be split across multiple arrays, create multiple params objects
    if layerParams[j]['type'] in ('conv','dense'):
        
        # Number of total rows used in MVM
        if layerParams[j]['type'] == 'conv':
            Nrows = layerParams[j]['Kx']*layerParams[j]['Ky']*layerParams[j]['Nic']
            Ncols = layerParams[j]['Noc']
            if not config.digital_bias:
                if layerParams[j]['bias'] or config.fold_batchnorm:
                    Nrows += 1

        elif layerParams[j]['type'] == 'dense':
            Nrows = sizes[j][2]
            Ncols = sizes[j+1][2]
            if not config.digital_bias and layerParams[j]['bias']:
                Nrows += 1

        # Compute number of arrays matrix must be partitioned across
        if config.NrowsMax > 0:
            Ncores = (Nrows-1)//config.NrowsMax + 1
        else:
            Ncores = 1

        if config.NcolsMax > 0:
            Ncores *= (Ncols-1)//config.NcolsMax + 1
        else:
            Ncores *= 1

       # Layer specific ADC and activation resolution and range (set above)
        adc_range = adc_ranges[j_mvm]
        if adc_range is not None:
            adc_range = adc_range.tolist()
        dac_range = dac_ranges[j_mvm]
        adc_bits_j = config.adc_bits_vec[j_mvm]
        dac_bits_j = config.dac_bits_vec[j_mvm]
        if config.gate_input:
            Rp_j = config.Rp_col
        else:
            Rp_j = np.maximum(config.Rp_row,config.Rp_col)

        # If parasitics are enabled, x_par and y_par are modified to optimize cumulative sum runtime
        if Rp_j > 0 and layerParams[j]['type'] == 'conv':
            xy_pars[j_conv,:] = get_xy_parallel_parasitics(Nrows,sizes[j][0],sizes[j+1][0],config.model_name,
                disable=config.disable_SW_packing)

        if layerParams[j]['type'] == 'conv':
            x_par, y_par = xy_pars[j_conv,:]
            convParams = layerParams[j]

        elif layerParams[j]['type'] == 'dense':
            x_par, y_par = 1, 1
            convParams = None

        # Does this layer use analog batchnorm?
        analog_batchnorm = config.fold_batchnorm and layerParams[j]['batch_norm'] is not None

        # Whether to profile ADC inputs inside the core in this layer
        # Currently this is supported only if NEITHER row or col partitioning is enabled
        profile_ADC_reluAware_j = False
        if profile_ADC_reluAware and Ncores == 1:
            if layerParams[j]['activation'] is not None and layerParams[j]['activation']['type'] == "RECTLINEAR":
                profile_ADC_reluAware_j = True

        params = set_params(task=config.task,
            wtmodel=config.style,
            convParams=convParams,
            alpha_noise=config.alpha_noise,
            balanced_style=config.balanced_style,
            ADC_per_ibit=config.ADC_per_ibit,
            x_par=x_par,
            y_par=y_par,
            weight_bits=config.weight_bits,
            weight_percentile=config.weight_percentile,
            useGPU=config.useGPU,
            proportional_noise=config.proportional_noise,
            alpha_error=config.alpha_error,
            adc_bits=adc_bits_j,
            dac_bits=dac_bits_j,
            adc_range=adc_range,
            dac_range=dac_range,
            error_model=config.error_model,
            noise_model=config.noise_model,
            NrowsMax=config.NrowsMax,
            NcolsMax=config.NcolsMax,
            positiveInputsOnly=positiveInputsOnly[j_mvm],
            input_bitslicing=config.input_bitslicing,
            gate_input=config.gate_input,
            subtract_current_in_xbar=config.subtract_current_in_xbar,
            interleaved_posneg=config.interleaved_posneg,
            t_drift=config.t_drift,
            drift_model=config.drift_model,
            Rp_row=config.Rp_row,
            Rp_col=config.Rp_col,
            digital_offset=config.digital_offset,
            Icol_max=config.Icol_max/config.Icell_max,
            infinite_on_off_ratio=config.infinite_on_off_ratio,
            Rmin = config.Rmin,
            Rmax = config.Rmax,
            adc_range_option=config.adc_range_option,
            proportional_error=config.proportional_error,
            Nslices=config.Nslices,
            digital_bias=config.digital_bias,
            analog_batchnorm=analog_batchnorm,
            adc_type=config.adc_type,
            profile_ADC_inputs=profile_ADC_inputs,
            profile_ADC_reluAware=profile_ADC_reluAware_j)

        if Ncores == 1:
            paramsList[j] = params
        else:
            paramsList[j] = Ncores*[None]
            for k in range(Ncores):
                paramsList[j][k] = params.copy()            
        
        j_mvm += 1
        if layerParams[j]['type'] == 'conv':
            j_conv += 1
        nrow.append(Nrows)
     # Need to make a copy to prevent inference() from modifying layerParams
    layerParamsCopy[j] = layerParams[j].copy()
   

fisrt_conv_dense = 0
for j in range(Nlayers):
    # For a layer that must be split across multiple arrays, create multiple params objects
    if layerParams[j]['type'] in ('conv','dense'):
        fisrt_conv_dense = j 
        break
# Run inference
accuracy, _ ,layerNums,ncoresList= inference(ntest=config.ntest,
    dataset=config.task,
    paramsList=paramsList,
    sizes=sizes,
    onnx_model=onnx_model,
    layerParams=layerParamsCopy,
    quantize = quantize,
    weight_for_quantized = weight_for_quantized,
    fisrt_conv_dense=fisrt_conv_dense,
    search_accuracy = False,
    config=config,
    useGPU=config.useGPU,
    count_interval=config.count_interval,
    randomSampling=config.randomSampling,
    topk=config.topk,
    subtract_pixel_mean=config.subtract_pixel_mean,
    memory_window=config.memory_window,
    model_name=config.model_name,
    fold_batchnorm=config.fold_batchnorm,
    digital_bias=config.digital_bias,
    nstart=config.nstart,
    ntest_batch=config.ntest_batch,
    bias_bits=config.bias_bits,
    time_interval=config.time_interval,
    imagenet_preprocess=config.imagenet_preprocess,
    dataset_normalization=config.dataset_normalization,
    adc_range_option=config.adc_range_option,
    show_HW_config=config.show_HW_config,
    calibration=calibration,
    profiling_folder=profiling_folder,
    profiling_settings=[profile_DAC_inputs,profile_ADC_inputs,profile_ADC_reluAware])
print(layerNums,ncoresList)
print(nrow)
print(relu_list)