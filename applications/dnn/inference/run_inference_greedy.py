#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#



import numpy as np
import os, sys, pickle, argparse,random
import pandas as pd
#To import parameters
sys.path.append("../../../simulator/")
#To import simulator
sys.path.append("../../../")
#To import dataset loader
sys.path.append("../")
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"

from interface.inference_net import set_params, inference
from interface.keras_parser import get_keras_metadata
from interface.onnx_parser import get_onnx_metadata
from interface.dnn_setup import augment_parameters, build_keras_model, model_specific_parameters, \
    get_xy_parallel, get_xy_parallel_parasitics, load_adc_activation_ranges
from interface.config_message import print_configuration_message

# ==========================
# ==== Load config file ====
# ==========================

import inference_config as config
import onnx
import ast
import os

navcim_dir = os.getenv('NAVCIM_DIR')
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--model', default='VGG16', help='VGG16|ResNet50|NasNetA|LFFD')
parser.add_argument('--ntest', type=int, default=200)
parser.add_argument('--ntest_batch', type=int, default=200)
parser.add_argument('--Nslices', type=int, default=1, help='Nslices')
parser.add_argument('--adc_bits', type=int, default=8, help='adc_bits')
parser.add_argument('--cell_bits', type=int, default=7, help='cell_bits')
parser.add_argument('--search_adc_range', type=bool, default=False, help='search_adc_range')
parser.add_argument('--metric', type=str, default='cka', choices=['mse', 'cosine', 'ssim','cka','hamming'], help='metric')
parser.add_argument('--candidate1', type=str, default=None, help='NrowsMax')
parser.add_argument('--candidate2', type=str, default=None, help='NrowsMax')

args = parser.parse_args()


config.model_name = args.model
config.ntest = args.ntest 
config.ntest_batch = args.ntest_batch
config.adc_bits = args.adc_bits
config.searchADC = args.search_adc_range



quantize = False
if config.model_name== 'MobileNetV2':
    onnx_model = onnx.load(f"{navcim_dir}/cross-sim/applications/dnn/inference/model/MobileNetV2.onnx")
    quantize =  True
elif config.model_name== 'ResNet50':
    onnx_model = onnx.load(f"{navcim_dir}/cross-sim/applications/dnn/inference/model/ResNet50.onnx")
    quantize =  True
elif config.model_name == 'SqueezeNet':
    onnx_model = onnx.load(f"{navcim_dir}/cross-sim/applications/dnn/inference/model/SqueezeNet.onnx")
else:
    raise ValueError("Invalid model name")

config.Nslices = np.ceil((config.weight_bits-1) / args.cell_bits).astype(int)
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




layerParams, sizes ,weight_for_quantized,output_node_name= get_onnx_metadata(onnx_model,config.model_name ,quantize, debug_graph=False, search_accuracy =True)


Nlayers = len(layerParams)
config.Nlayers_mvm = np.sum([(layerParams[j]['type'] in ('conv','dense')) for j in range(Nlayers)])

config.output_node_name = output_node_name
# ===================================================================
# ======= Parameter validation and model-specific parameters ========
# ===================================================================
# General parameter check
config = augment_parameters(config)
# Parameters specific to some neural networks
config, positiveInputsOnly = model_specific_parameters(config)

# =========================================================
# ==== Load calibrated ranges for ADCs and activations ====
# =========================================================



adc_ranges_cadidate = []
index_row_col = []
if args.candidate1 is None:
    Row_size_candidate = [64,128,256]
    Col_size_candidate = [64,128,256]


    for row in Row_size_candidate:
        for col in Col_size_candidate:
            adc_ranges_tmp, dac_ranges, positiveInputsOnly= load_adc_activation_ranges(config,positiveInputsOnly, NrowsMax_ = row, NcolsMax_=col)
            index_row_col.append([row,col])
            adc_ranges_cadidate.append(adc_ranges_tmp)
else:
    array1 = ast.literal_eval(args.candidate1)
    index_row_col.append(array1)
    if args.candidate2 is not None:
        array2 = ast.literal_eval(args.candidate2)
        index_row_col.append(array2)

    for i in index_row_col:
        adc_ranges_tmp, dac_ranges, positiveInputsOnly= load_adc_activation_ranges(config,positiveInputsOnly, NrowsMax_ = i[0], NcolsMax_=i[1])
        adc_ranges_cadidate.append(adc_ranges_tmp)


# adc_ranges_cadidate[0] = adc_ranges_cadidate[1]
# adc_ranges_cadidate[1] = adc_ranges_cadidate[0]
print("adc_ranges_cadidate",adc_ranges_cadidate)
# quit()


 


# =======================================
# ======= GPU performance tuning ========
# =======================================

# Convolutions: number of sliding windows along x and y to compute in parallel
xy_pars = get_xy_parallel(config, disable=config.disable_SW_packing)

# ================================
# ========= Start sweep ==========
# ================================

# Display the chosen simulation settings
print_configuration_message(config)


def calculate_fitness_with_simulation(individual,adc_idx,search_accuracy=False, validate_mse=False):
    print("test", individual)
    for ind in range(len(individual)):
        if individual[ind] == -1 or individual[ind] == None:
            individual[ind] = 0


    for ind in range(len(adc_idx)):
        if adc_idx[ind] == -1 or adc_idx[ind] == None or args.search_adc_range==False: 
            adc_idx[ind] = 0
    if config.Nruns > 1:
        print('')
        print('===========')
        print(" Run "+str(q+1)+"/"+str(config.Nruns))
        print('===========')
    
    paramsList, layerParamsCopy = Nlayers*[None], Nlayers*[None]
    j_mvm, j_conv ,j_mvm_wo_depthwise= 0, 0 , 0# counter for MVM and conv layers
    mvm_to_n = {}
    n_to_mvm = {}
    # ===================================================
    # ==== Compute and set layer-specific parameters ====
    # ===================================================
    
    for j in range(Nlayers):

        # For a layer that must be split across multiple arrays, create multiple params objects
        if layerParams[j]['type'] in ('conv','dense'):
            mvm_to_n[j_mvm] = j
            n_to_mvm[j] = j_mvm
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

            NrowsMax = index_row_col[individual[j_mvm]][0]
            NcolsMax = index_row_col[individual[j_mvm]][1]
            config.NrowsMax = NrowsMax
            config.NcolsMax = NcolsMax
            adc_ranges = adc_ranges_cadidate[individual[j_mvm]]
     
            if NrowsMax > 0:
                Ncores = (Nrows-1)//NrowsMax + 1
            else:
                Ncores = 1

            if NcolsMax > 0:
                Ncores *= (Ncols-1)//NcolsMax + 1

            # Layer specific ADC and activation resolution and range (set above)
            adc_range = adc_ranges[j_mvm_wo_depthwise][adc_idx[j_mvm_wo_depthwise]]
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

            if config.Icol_max > 0:
                Icol_max_norm = config.Icol_max / config.Icell_max
            else:
                Icol_max_norm = 0

            # Does this layer use analog batchnorm?
            analog_batchnorm = config.fold_batchnorm and layerParams[j]['batch_norm'] is not None

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
                NrowsMax=NrowsMax,
                NcolsMax=NcolsMax,
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
                Icol_max=Icol_max_norm,
                infinite_on_off_ratio=config.infinite_on_off_ratio,
                Rmin = config.Rmin,
                Rmax = config.Rmax,
                adc_range_option=config.adc_range_option,
                proportional_error=config.proportional_error,
                Nslices=config.Nslices,
                digital_bias=config.digital_bias,
                analog_batchnorm=analog_batchnorm,
                adc_type=config.adc_type,
                input_slice_size=config.input_slice_size)
            
    
            if Ncores == 1:
                paramsList[j] = params
            else:
                paramsList[j] = Ncores*[None]
                for k in range(Ncores):
                    paramsList[j][k] = params.copy()            
            
            j_mvm += 1
            if layerParams[j]['type'] == 'conv':
                j_conv += 1
            if layerParams[j]['type'] == 'dense' or (layerParams[j]['type'] == 'conv' and not layerParams[j]['depthwise']):
                j_mvm_wo_depthwise+= 1

         # Need to make a copy to prevent inference() from modifying layerParams
        layerParamsCopy[j] = layerParams[j].copy()


    fisrt_conv_dense = 0
    for j in range(Nlayers):
        # For a layer that must be split across multiple arrays, create multiple params objects
        if layerParams[j]['type'] in ('conv','dense'):
            fisrt_conv_dense = j 
            break
    # Run inference
    if search_accuracy and not validate_mse:
        best_mse_idx, mse_list = inference(ntest=config.ntest,
            dataset=config.task,
            paramsList=paramsList,
            sizes=sizes,
            onnx_model=onnx_model,
            layerParams=layerParamsCopy,
            quantize = quantize,
            weight_for_quantized = weight_for_quantized,
            fisrt_conv_dense=fisrt_conv_dense,
            search_accuracy = search_accuracy,
            config=config,
            mvm_to_n=mvm_to_n,
            n_to_mvm=n_to_mvm, 
            index_row_col=index_row_col,
            adc_ranges_cadidate=adc_ranges_cadidate,
            metric=args.metric,
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
            return_network_output=config.return_network_output)
        return best_mse_idx, mse_list
    elif search_accuracy and validate_mse: 
        best_mse_idx, mse_list = inference(ntest=config.ntest,
            dataset=config.task,
            paramsList=paramsList,
            sizes=sizes,
            onnx_model=onnx_model,
            layerParams=layerParamsCopy,
            quantize = quantize,
            weight_for_quantized = weight_for_quantized,
            fisrt_conv_dense=fisrt_conv_dense,
            search_accuracy = search_accuracy,
            config=config,
            mvm_to_n=mvm_to_n,
            n_to_mvm=n_to_mvm, 
            index_row_col=index_row_col,
            adc_ranges_cadidate=adc_ranges_cadidate,
            metric=args.metric,
            validate_mse=validate_mse,
            validate_seq=individual,
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
            return_network_output=config.return_network_output)
        return best_mse_idx, mse_list
    else:
        accuracy, network_output ,layerNums,ncoresList= inference(ntest=config.ntest,
            dataset=config.task,
            paramsList=paramsList,
            sizes=sizes,
            onnx_model=onnx_model,
            layerParams=layerParamsCopy,
            quantize = quantize,
            weight_for_quantized = weight_for_quantized,
            fisrt_conv_dense=fisrt_conv_dense,
            search_accuracy = search_accuracy,
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
            return_network_output=config.return_network_output)

        return accuracy , network_output






# Parameters
population_size = 10
individual_length = config.Nlayers_mvm  # Number of layers
generations = 10


init_idx = []
init_adc_idx = []
for j in range(Nlayers):
    if layerParams[j]['type'] in ('conv'):
        if layerParams[j]["depthwise"] is True:
            init_idx.append(None)
        else:
            init_idx.append(0)
            init_adc_idx.append(0)
    elif layerParams[j]['type'] in ('dense'):
        init_idx.append(0)
        init_adc_idx.append(0)




best_mse_idx, mse_list = calculate_fitness_with_simulation(init_idx,init_adc_idx,True)

if args.candidate1 is None:
    cleaned_data = [sublist for sublist in mse_list if not all(item[0] is None for item in sublist)]
    array_data = np.array([[item[0] for item in sublist if item[0] is not None] for sublist in cleaned_data], dtype=np.float32)
    col_name = [f'{i[0]},{i[1]}' for i in index_row_col]
    print(col_name)
    df = pd.DataFrame(array_data,columns=col_name)
    df.to_csv(f'{args.model}_{args.metric}_ADC:{args.adc_bits}_CellBit{args.cell_bits}_list.csv', index=False)

calculate_fitness_with_simulation(best_mse_idx[0],best_mse_idx[1],False)
print(mse_list)
# calculate_fitness_with_simulation([-1, None, -1, 2, None, 2, 2, None, 2, 2, None, 2, 2, None, 2, 2, None, 1, 2, None, 2, 2, None, 2, 1, None, 2, 2, None, 1, 0, None, 1, 0, None, 2, 0, None, 2, 0, None, 2, 0, None, 1, 0, None, 2, 0, None, 2, 0, 0],[0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, 0],False)

