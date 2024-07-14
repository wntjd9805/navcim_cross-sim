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
from interface.dnn_setup import augment_parameters, build_keras_model, model_specific_parameters, get_xy_parallel, get_xy_parallel_parasitics, load_adc_activation_ranges
from interface.config_message import print_configuration_message

# ==========================
# ==== Load config file ====
# ==========================

# import inference_config as config
import onnx
import ast
import os

navcim_dir = os.getenv('NAVCIM_DIR')

class InferenceConfig:
    def __init__(self, task="imagenet", useGPU=False, gpu_num=7, Nruns=1, ntest=1000, 
                 ntest_batch=1000, nstart=0, randomSampling=False, return_network_output=False,
                 show_model_summary=False, show_HW_config=False, disable_SW_packing=False,
                 weight_bits=8, weight_percentile=100, Nslices=1, NrowsMax=1152, NcolsMax=0,
                 style="BALANCED", balanced_style="ONE_SIDED", subtract_current_in_xbar=True,
                 digital_offset=True, Rp_row=0, Rp_col=0, gate_input=False, interleaved_posneg=False,
                 Icol_max=0, Icell_max=1.8e-6, fold_batchnorm=True, digital_bias=True, bias_bits=0,
                 Rmin=1e3, Rmax=1e6, infinite_on_off_ratio=False, error_model="none", alpha_error=0.00,
                 proportional_error=True, noise_model="generic", alpha_noise=0.00, proportional_noise=True,
                 t_drift=0, drift_model="none", adc_bits=5, dac_bits=8, input_bitslicing=False,
                 input_slice_size=1, ADC_per_ibit=False, adc_range_option="CALIBRATED", adc_type="generic",
                 input_framework='onnx', searchADC=False , count_interval=1, topk=(1,5),time_interval = True):
        self.useGPU = useGPU
        self.gpu_num = gpu_num
        self.Nruns = Nruns
        self.ntest = ntest
        self.ntest_batch = ntest_batch
        self.nstart = nstart
        self.randomSampling = randomSampling
        self.return_network_output = return_network_output
        self.show_model_summary = show_model_summary
        self.show_HW_config = show_HW_config
        self.disable_SW_packing = disable_SW_packing
        self.weight_bits = weight_bits
        self.weight_percentile = weight_percentile
        self.Nslices = Nslices
        self.NrowsMax = NrowsMax
        self.NcolsMax = NcolsMax
        self.style = style
        self.balanced_style = balanced_style
        self.subtract_current_in_xbar = subtract_current_in_xbar
        self.digital_offset = digital_offset
        self.Rp_row = Rp_row
        self.Rp_col = Rp_col
        self.gate_input = gate_input
        self.interleaved_posneg = interleaved_posneg
        self.Icol_max = Icol_max
        self.Icell_max = Icell_max
        self.fold_batchnorm = fold_batchnorm
        self.digital_bias = digital_bias
        self.bias_bits = bias_bits
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.infinite_on_off_ratio = infinite_on_off_ratio
        self.error_model = error_model
        self.alpha_error = alpha_error
        self.proportional_error = proportional_error
        self.noise_model = noise_model
        self.alpha_noise = alpha_noise
        self.proportional_noise = proportional_noise
        self.t_drift = t_drift
        self.drift_model = drift_model
        self.adc_bits = adc_bits
        self.dac_bits = dac_bits
        self.input_bitslicing = input_bitslicing
        self.input_slice_size = input_slice_size
        self.ADC_per_ibit = ADC_per_ibit
        self.adc_range_option = adc_range_option
        self.adc_type = adc_type
        self.input_framework = input_framework
        self.searchADC = searchADC
        self.task = task
        self.count_interval = count_interval
        self.topk = topk
        self.time_interval = time_interval

def load_config_from_file(config_file_path):
    # 설정 파일을 읽어서 Python 코드로 실행
    with open(config_file_path, 'r') as file:
        config_code = file.read()
    
    # 지역 변수로 InferenceConfig 클래스 인스턴스 생성
    local_vars = {
        'InferenceConfig': InferenceConfig,
    }
    exec(config_code, globals(), local_vars)
    
    # 인스턴스 생성
    config_instance = local_vars['InferenceConfig']()
    return config_instance

def init(model,ntest,ntest_batch,cell_bits,adc_bits):
    navcim_dir = os.getenv('NAVCIM_DIR')
    config = load_config_from_file(f'{navcim_dir}/cross-sim/applications/dnn/inference/inference_config.py')

    config.model_name = model
    config.ntest = ntest 
    config.ntest_batch = ntest_batch
    config.Nslices = np.ceil((config.weight_bits-1) / cell_bits).astype(int)
    config.adc_bits = adc_bits
    # ===================
    # ==== GPU setup ====
    # ===================

    # Restrict tensorflow GPU memory usage
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(-1)
    # import tensorflow as tf
    # gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    # for k in range(len(gpu_devices)):
    #     tf.config.experimental.set_memory_growth(gpu_devices[k], True)

    # # Set up GPU
    # if config.useGPU:
    #     import cupy
    #     os.environ["CUDA_VISIBLE_DEVICES"]=str(config.gpu_num)
    #     os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #     cupy.cuda.Device(0).use()

    # =====================================
    # ==== Import neural network model ====
    # =====================================

    # Build Keras model and prepare CrossSim-compatible topology metadata
    # keras_model = build_keras_model(config.model_name,show_model_summary=config.show_model_summary)
    # layerParams, sizes = get_keras_metadata(keras_model,task=config.task,debug_graph=False)
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


    layerParams, sizes ,weight_for_quantized,output_node_name = get_onnx_metadata(onnx_model,config.model_name ,quantize, debug_graph=False, search_accuracy =False)
    Nlayers = len(layerParams)
    config.Nlayers_mvm = np.sum([(layerParams[j]['type'] in ('conv','dense')) for j in range(Nlayers)])
    config.output_node_name = output_node_name
    # ===================================================================
    # ======= Parameter validation and model-specific parameters ========
    # ===================================================================
    # General parameter check
    config = augment_parameters(config)
    config, positiveInputsOnly = model_specific_parameters(config)

    # =========================================================
    # ==== Load calibrated ranges for ADCs and activations ====
    # =========================================================

    Row_size_candidate = [64,128,256]
    Col_size_candidate = [64,128,256]
    adc_ranges_cadidate = []
    index_row_col = []
    for row in Row_size_candidate:
        for col in Col_size_candidate:
            adc_ranges_tmp, dac_ranges, positiveInputsOnly= load_adc_activation_ranges(config,positiveInputsOnly, NrowsMax_ = row, NcolsMax_=col)
            index_row_col.append([row,col])
            adc_ranges_cadidate.append(adc_ranges_tmp)

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

    return layerParams, sizes, onnx_model, quantize, adc_ranges_cadidate, index_row_col, dac_ranges, positiveInputsOnly, xy_pars, weight_for_quantized,config


def calculate_fitness_with_simulation(individual,adc_idx,layerParams, sizes, onnx_model, quantize, adc_ranges_cadidate, index_row_col, dac_ranges, positiveInputsOnly, xy_pars, weight_for_quantized,config):
    print("test", individual)
    for ind in range(len(individual)):
        if individual[ind] == -1 or individual[ind] == None:
            individual[ind] = 0
    for ind in range(len(adc_idx)):
        if adc_idx[ind] == -1 or adc_idx[ind] == None: 
            adc_idx[ind] = 0
    if config.Nruns > 1:
        print('')
        print('===========')
        print(" Run "+str(q+1)+"/"+str(config.Nruns))
        print('===========')
    
    Nlayers = len(layerParams)
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
    accuracy, network_output ,layerNums,ncoresList= inference(ntest=config.ntest,
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
        return_network_output=config.return_network_output)

    return accuracy




# # 입력 문자열에서 리스트 부분과 시퀀스 부분 분리

# model = 'MobileNetV2'
# ntest = 10
# ntest_batch = 10
# Nslices = 1
# adc_bits = 8

# layerParams, sizes, onnx_model, quantize, adc_ranges_cadidate, index_row_col, dac_ranges, positiveInputsOnly, xy_pars, weight_for_quantized,config = init(model,ntest,ntest_batch,Nslices,adc_bits)


# print(config)
# input_list_str, sequence_str = args.input.split('_')[0], args.input.split('_')[1:]

# # 문자열 형태의 리스트를 실제 파이썬 리스트로 변환
# input_list = ast.literal_eval(input_list_str)

# # 시퀀스 문자열을 정수 리스트로 변환
# sequence = [int(num) for num in sequence_str]

# print(input_list)
# print(index_row_col)
# print(index_row_col.index([128,128]))
# # 시퀀스에 따라 리스트 재구성
# reconstructed_list = [index_row_col.index([input_list[num][0],input_list[num][1]]) for num in sequence]

# init_idx = []
# init_adc_idx = []
# mvm = 0
# Nlayers = len(layerParams)
# for j in range(Nlayers):
#     if layerParams[j]['type'] in ('conv'):
#         if layerParams[j]["depthwise"] is True:
#             init_idx.append(None)
#         else:
#             init_idx.append(reconstructed_list[mvm])
#             init_adc_idx.append(0)
#             mvm += 1
#     elif layerParams[j]['type'] in ('dense'):
#         init_idx.append(reconstructed_list[mvm])
#         init_adc_idx.append(0)
#         mvm += 1


# calculate_fitness_with_simulation(init_idx,init_adc_idx,layerParams, sizes, onnx_model, quantize, adc_ranges_cadidate, index_row_col, dac_ranges, positiveInputsOnly, xy_pars, weight_for_quantized,config)
# calculate_fitness_with_simulation(init_idx,init_adc_idx,layerParams, sizes, onnx_model, quantize, adc_ranges_cadidate, index_row_col, dac_ranges, positiveInputsOnly, xy_pars, weight_for_quantized,config)

# calculate_fitness_with_simulation([-1, None, -1, 8, None, 8, 8, None, 8, 8, None, 8, 8, None, 8, 2, None, 3, 8, None, 8, 0, None, 6, 6, None, 8, 7, None, 4, 2, None, 4, 2, None, 8, 1, None, 6, 2, None, 7, 0, None, 4, 0, None, 7, 5, None, 7, 0, 0],[0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, 0],False)
# quit()
