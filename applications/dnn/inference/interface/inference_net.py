#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import sys, io, os, time, random
import numpy as np
import tensorflow.keras.backend as K
from simulator import DNN, CrossSimParameters
from dataset_loaders import load_dataset_inference
from helpers.qnn_adjustment import qnn_adjustment
from simulator.parameters.core_parameters import CoreStyle, BitSlicedCoreStyle
import onnx
from onnx import numpy_helper

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def set_params(**kwargs):
    """
    Pass parameters using kwargs to allow for a general parameter dict to be used
    This function should be called before train and sets all parameters of the neural core simulator

    :return: params, a parameter object with all the settings

    """
    #######################
    #### load relevant parameters from arg dict
    task = kwargs.get("task","small")
    wtmodel = kwargs.get("wtmodel","BALANCED")
    convParams = kwargs.get("convParams",None)

    error_model = kwargs.get("error_model","none")
    alpha_error = kwargs.get("alpha_error",0.0)
    proportional_error = kwargs.get("proportional_error",False)

    noise_model = kwargs.get("noise_model","none")
    alpha_noise = kwargs.get("alpha_noise",0.0)
    proportional_noise = kwargs.get("proportional_noise",False)

    t_drift = kwargs.get("t_drift",0)
    drift_model = kwargs.get("drift_model",None)

    Rp_row = kwargs.get("Rp_row",0)
    Rp_col = kwargs.get("Rp_col",0)

    NrowsMax = kwargs.get("NrowsMax",None)
    NcolsMax = kwargs.get("NcolsMax",None)
    weight_bits = kwargs.get("weight_bits",0)
    weight_percentile = kwargs.get("weight_percentile",100)
    adc_bits = kwargs.get("adc_bits",8)
    dac_bits = kwargs.get("dac_bits",8)
    dac_range = kwargs.get("dac_range",(0,1))
    adc_range = kwargs.get("adc_range",(0,1))
    adc_type = kwargs.get("adc_type","generic")
    positiveInputsOnly = kwargs.get("positiveInputsOnly",False)
    interleaved_posneg = kwargs.get("interleaved_posneg",False)
    subtract_current_in_xbar = kwargs.get("subtract_current_in_xbar",True)
    Rmin = kwargs.get("Rmin", 1000)
    Rmax = kwargs.get("Rmax", 10000)
    infinite_on_off_ratio = kwargs.get("infinite_on_off_ratio", True)
    gate_input = kwargs.get("gate_input",True)

    Nslices = kwargs.get("Nslices",1)    
    digital_offset = kwargs.get("digital_offset",False)
    adc_range_option = kwargs.get("adc_range_option",False)
    Icol_max = kwargs.get("Icol_max",1e6)
    digital_bias = kwargs.get("digital_bias",False)
    analog_batchnorm = kwargs.get("analog_batchnorm",False)

    x_par = kwargs.get("x_par",1)
    y_par = kwargs.get("y_par",1)
    weight_reorder = kwargs.get("weight_reorder",True)
    conv_matmul = kwargs.get("conv_matmul",True)
    useGPU = kwargs.get("useGPU",False)
    gpu_id = kwargs.get("gpu_id",0)
    profile_ADC_inputs = kwargs.get("profile_ADC_inputs",False)
    profile_ADC_reluAware = kwargs.get("profile_ADC_reluAware",False)

    balanced_style = kwargs.get("balanced_style","one_sided")
    input_bitslicing = kwargs.get("input_bitslicing",False)
    input_slice_size = kwargs.get("input_slice_size",1)
    ADC_per_ibit = kwargs.get("ADC_per_ibit",False)
    disable_parasitics_slices = kwargs.get("disable_parasitics_slices",None)

    ################  create parameter objects with all core settings
    params = CrossSimParameters()

    params.dac_range = dac_range
    params.convParams = convParams
    params.positiveInputsOnly = positiveInputsOnly
    ############### Numerical simulation settings
    params.simulation.useGPU = useGPU
    if useGPU:
        params.simulation.gpu_id = gpu_id

    ### Weight reoder: Whether to reorder the duplicated weights into Toeplitz form when x_par, y_par > 1 so that the
    # expanded matrix takes up less memory
    if convParams is None:
        weight_reorder = False
    else:
        # This option is disabled if any of the following conditions are True
        # 1-3) No reuse to exploit
        noWR_cond1 =    (x_par == 1 and y_par == 1)
        noWR_cond2 =    (convParams['Kx'] == 1 and convParams['Ky'] == 1)
        noWR_cond3 =    ((x_par > 1 and convParams['stride'] >= convParams['Kx']) and \
                        (y_par > 1 and convParams['stride'] >= convParams['Ky']))
        # 4-11) Cases that don't make sense to implement (as of now)
        noWR_cond4 =    (NrowsMax < convParams['Kx']*convParams['Ky']*convParams['Nic'])
        noWR_cond5 =    (convParams['bias'] and not digital_bias)
        noWR_cond6 =    (analog_batchnorm and not digital_bias)
        noWR_cond7 =    (wtmodel == "OFFSET")
        noWR_cond8 =    (Rp_col > 0)
        noWR_cond9 =    (Rp_row > 0 and not gate_input)
        noWR_cond10 =   (noise_model == "generic" and alpha_noise > 0)
        noWR_cond11 =   (noise_model != "none" and noise_model != "generic")

        if any([noWR_cond1, noWR_cond2, noWR_cond3, noWR_cond4, noWR_cond5, noWR_cond6, 
            noWR_cond7, noWR_cond8, noWR_cond9, noWR_cond10, noWR_cond11]):
            weight_reorder = False

    # Enable conv matmul?
    if convParams is None:
        conv_matmul = False
    else:
        # These cases cannot be realistically modeled with matmul
        noCM_cond1 = (Rp_col > 0)
        noCM_cond2 = (Rp_row > 0 and not gate_input)
        noCM_cond3 = (noise_model == "generic" and alpha_noise > 0)
        noCM_cond4 = (noise_model != "none" and noise_model != "generic")
        if any([noCM_cond1, noCM_cond2, noCM_cond3, noCM_cond4]):
            conv_matmul = False

    if conv_matmul:
        weight_reorder = False
        x_par = 1
        y_par = 1

    # Multiple convolutional MVMs in parallel? (only used if conv_matmul = False)
    params.simulation.convolution.x_par = int(x_par) # Number of sliding window steps to do in parallel (x)
    params.simulation.convolution.y_par = int(y_par) # Number of sliding window steps to do in parallel (y)
    params.simulation.convolution.weight_reorder = weight_reorder
    params.simulation.convolution.conv_matmul = conv_matmul

    ############### Crossbar weight mapping settings

    if Nslices == 1:
        params.core.style = wtmodel
    else:
        params.core.style = "BITSLICED"
        params.core.bit_sliced.style = wtmodel

    if NrowsMax is not None:
        params.core.rows_max = NrowsMax

    if NcolsMax is not None:
        params.core.cols_max = NcolsMax

    params.core.balanced.style = balanced_style
    params.core.balanced.subtract_current_in_xbar = subtract_current_in_xbar
    if digital_offset:
        params.core.offset.style = "DIGITAL_OFFSET"
    else:
        params.core.offset.style = "UNIT_COLUMN_SUBTRACTION"

    params.xbar.device.Rmin = Rmin
    params.xbar.device.Rmax = Rmax
    params.xbar.device.infinite_on_off_ratio = infinite_on_off_ratio

    ############### Convolution

    if convParams is not None:
        params.simulation.convolution.is_conv_core = (convParams['type'] == 'conv')
        params.simulation.convolution.stride = convParams['stride']
        params.simulation.convolution.Kx = convParams['Kx']
        params.simulation.convolution.Ky = convParams['Ky']
        params.simulation.convolution.Noc = convParams['Noc']
        params.simulation.convolution.Nic = convParams['Nic']

    ############### Device errors

    #### Read noise
    if noise_model == "generic" and alpha_noise > 0:
        params.xbar.device.read_noise.enable = True
        params.xbar.device.read_noise.magnitude = alpha_noise
        if not proportional_noise:
            params.xbar.device.read_noise.model = "NormalIndependentDevice"
        else:
            params.xbar.device.read_noise.model = "NormalProportionalDevice"
    elif noise_model != "generic" and noise_model != "none":
        params.xbar.device.read_noise.enable = True
        params.xbar.device.read_noise.model = noise_model

    ##### Programming error
    if error_model == "generic" and alpha_error > 0:
        params.xbar.device.programming_error.enable = True
        params.xbar.device.programming_error.magnitude = alpha_error
        if not proportional_error:
            params.xbar.device.programming_error.model = "NormalIndependentDevice"
        else:
            params.xbar.device.programming_error.model = "NormalProportionalDevice"
    elif error_model != "generic" and error_model != "none":
        params.xbar.device.programming_error.enable = True
        params.xbar.device.programming_error.model = error_model

    # Drift
    if drift_model != "none":
        params.xbar.device.drift_error.enable = True
        params.xbar.device.time = t_drift
        params.xbar.device.drift_error.model = drift_model

    ############### Parasitic resistance

    if Rp_col > 0 or Rp_row > 0:
        # Bit line parasitic resistance
        params.xbar.array.parasitics.enable = True
        params.xbar.array.parasitics.Rp_col = Rp_col/Rmin
        params.xbar.array.parasitics.Rp_row = Rp_row/Rmin
        params.xbar.array.parasitics.gate_input = gate_input

        if gate_input and Rp_col == 0:
            params.xbar.array.parasitics.enable = False

        # Numeric params related to parasitic resistance simulation
        params.simulation.Niters_max_parasitics = 100
        params.simulation.Verr_th_mvm = 2e-4
        params.simulation.relaxation_gamma = 0.9 # under-relaxation

    ############### Weight bit slicing

    # Compute the number of cell bits
    if wtmodel == "OFFSET" and Nslices == 1:
        cell_bits = weight_bits
    elif wtmodel == "BALANCED" and Nslices == 1:
        cell_bits = weight_bits - 1
    elif Nslices > 1:
        # For weight bit slicing, quantization is done during mapping and does
        # not need to be applied at the xbar level
        if weight_bits % Nslices == 0:
            cell_bits = int(weight_bits / Nslices)
        elif wtmodel == "BALANCED":
            cell_bits = int(np.ceil((weight_bits-1)/Nslices))
        else:
            cell_bits = int(np.ceil(weight_bits/Nslices))
        params.core.bit_sliced.num_slices = Nslices
        if disable_parasitics_slices is not None:
            params.xbar.array.parasitics.disable_slices = disable_parasitics_slices
        else:
            params.xbar.array.parasitics.disable_slices = [False]*Nslices

    # Weights
    params.core.weight_bits = int(weight_bits)
    params.core.mapping.weights.percentile = weight_percentile/100

    params.xbar.device.cell_bits = cell_bits
    params.xbar.adc.mvm.adc_per_ibit = ADC_per_ibit
    params.xbar.adc.mvm.adc_range_option = adc_range_option
    params.core.balanced.interleaved_posneg = interleaved_posneg
    params.simulation.analytics.profile_adc_inputs = (profile_ADC_inputs and not profile_ADC_reluAware)
    params.xbar.array.Icol_max = Icol_max

    ###################### DAC settings

    params.xbar.dac.mvm.bits = int(dac_bits)
    if positiveInputsOnly:
        params.xbar.dac.mvm.model = "QuantizerDAC"
        params.xbar.dac.mvm.signed = False
    else:
        params.xbar.dac.mvm.model = "SignMagnitudeDAC"
        params.xbar.dac.mvm.signed = True

    if dac_bits > 0:
        params.xbar.dac.mvm.input_bitslicing = input_bitslicing
        params.xbar.dac.mvm.slice_size = input_slice_size
        if not digital_bias:
            dac_range[1] = np.maximum(dac_range[1],1)

        if not positiveInputsOnly:        
            if input_bitslicing or params.xbar.dac.mvm.model == "SignMagnitudeDAC":
                max_dac_range = float(np.max(np.abs(dac_range)))
                params.core.mapping.inputs.mvm.min = -max_dac_range
                params.core.mapping.inputs.mvm.max = max_dac_range
            else:
                params.core.mapping.inputs.mvm.min = float(dac_range[0])
                params.core.mapping.inputs.mvm.max = float(dac_range[1])
        else:
            params.core.mapping.inputs.mvm.min = float(dac_range[0])
            params.core.mapping.inputs.mvm.max = float(dac_range[1])
    else:
        params.xbar.dac.mvm.input_bitslicing = False
        params.core.mapping.inputs.mvm.min = -1e10
        params.core.mapping.inputs.mvm.max = 1e10

    ###################### ADC settings

    params.xbar.adc.mvm.bits = int(adc_bits)

    # Custom ADCs are currently set to ideal. Modify the parameters
    # below to model non-ideal ADC implementations
    if adc_type == "ramp":
        params.xbar.adc.mvm.model = "RampADC"
        params.xbar.adc.mvm.gain_db = 100
        params.xbar.adc.mvm.sigma_capacitor = 0.00
        params.xbar.adc.mvm.sigma_comparator = 0.00
        params.xbar.adc.mvm.symmetric_cdac = True
    elif adc_type == "sar" or adc_type == "SAR":
        params.xbar.adc.mvm.model = "SarADC"
        params.xbar.adc.mvm.gain_db = 100
        params.xbar.adc.mvm.sigma_capacitor = 0.00
        params.xbar.adc.mvm.sigma_comparator = 0.00
        params.xbar.adc.mvm.split_cdac = True
        params.xbar.adc.mvm.group_size = 8
    elif adc_type == "pipeline":
        params.xbar.adc.mvm.model = "PipelineADC"
        params.xbar.adc.mvm.gain_db = 100
        params.xbar.adc.mvm.sigma_C1 = 0.00
        params.xbar.adc.mvm.sigma_C2 = 0.00
        params.xbar.adc.mvm.sigma_Cpar = 0.00
        params.xbar.adc.mvm.sigma_comparator = 0.00
        params.xbar.adc.mvm.group_size = 8
    elif adc_type == "cyclic":
        params.xbar.adc.mvm.model = "CyclicADC"
        params.xbar.adc.mvm.gain_db = 100
        params.xbar.adc.mvm.sigma_C1 = 0.00
        params.xbar.adc.mvm.sigma_C2 = 0.00
        params.xbar.adc.mvm.sigma_Cpar = 0.00
        params.xbar.adc.mvm.sigma_comparator = 0.00
        params.xbar.adc.mvm.group_size = 8

    # Determine if signed ADC
    if adc_range_option == "CALIBRATED" and adc_bits > 0:
        params.xbar.adc.mvm.signed = bool(np.min(adc_range) < 0)
    else:
        params.xbar.adc.mvm.signed = (wtmodel == "BALANCED" or (wtmodel == "OFFSET" and not positiveInputsOnly))

    # Set the ADC model and the calibrated range
    if adc_bits > 0:
        if Nslices > 1:
            if adc_range_option == "CALIBRATED":
                params.xbar.adc.mvm.calibrated_range = adc_range
            if params.xbar.adc.mvm.signed and adc_type == "generic":
                params.xbar.adc.mvm.model = "SignMagnitudeADC"
            elif not params.xbar.adc.mvm.signed and adc_type == "generic":
                params.xbar.adc.mvm.model = "QuantizerADC"
        else:
            # This criterion checks if the center point of the range is within 1 level of zero
            # If that is the case, the range is made symmetric about zero and sign bit is used
            if adc_range_option == "CALIBRATED":
                if np.abs(0.5*(adc_range[0] + adc_range[1])/(adc_range[1] - adc_range[0])) < 1/pow(2,adc_bits):
                    absmax = np.max(np.abs(adc_range))
                    params.xbar.adc.mvm.calibrated_range = np.array([-absmax,absmax])
                    if adc_type == "generic":
                        params.xbar.adc.mvm.model = "SignMagnitudeADC"
                else:
                    params.xbar.adc.mvm.calibrated_range = adc_range
                    if adc_type == "generic":
                        params.xbar.adc.mvm.model = "QuantizerADC"

            elif adc_range_option == "MAX" or adc_range_option == "GRANULAR":
                if params.xbar.adc.mvm.signed and adc_type == "generic":
                    params.xbar.adc.mvm.model = "SignMagnitudeADC"
                elif not params.xbar.adc.mvm.signed and adc_type == "generic":
                    params.xbar.adc.mvm.model = "QuantizerADC"
                    
    return params


def inference(ntest,dataset,paramsList,sizes,onnx_model,layerParams,quantize,weight_for_quantized,fisrt_conv_dense,search_accuracy,config,mvm_to_n=None,n_to_mvm=None, index_row_col=None,adc_ranges_cadidate=None,metric=None ,validate_mse=False, validate_seq = None, **kwargs):
    """
    Runs inference on a full neural network whose weights are passed in as a Keras model and whose topology is specified by layerParams
    Parameters for individual neural cores generated using set_params() are passed in as paramsList
    """
    ########## load values from dict
    seed = random.randrange(1,1000000)
    
    count_interval = kwargs.get("count_interval",10)
    time_interval = kwargs.get("time_interval",False)
    randomSampling = kwargs.get("randomSampling",False)
    topk = kwargs.get("topk",1)
    nstart = kwargs.get("nstart",0)
    ntest_batch = kwargs.get("ntest_batch",0)
    calibration = kwargs.get("calibration",False)
    imagenet_preprocess = kwargs.get("imagenet_preprocess","cv2")
    dataset_normalization = kwargs.get("dataset_normalization","none")
    gpu_id = kwargs.get("gpu_id",0)
    useGPU = kwargs.get("useGPU",False)
    
    memory_window = kwargs.get("memory_window",0)
    subtract_pixel_mean = kwargs.get("subtract_pixel_mean",False)
    profiling_folder = kwargs.get("profiling_folder",None)
    profiling_settings = kwargs.get("profiling_settings",[False,False,False])
    model_name = kwargs.get("model_name",None)

    fold_batchnorm = kwargs.get("fold_batchnorm",False)
    batchnorm_style = kwargs.get("batchnorm_style","sqrt") # sqrt / no_sqrt
    digital_bias = kwargs.get("digital_bias",False)
    bias_bits = kwargs.get("bias_bits",0)
    show_HW_config = kwargs.get("show_HW_config",False)
    return_network_output = kwargs.get("return_network_output",False)

    ####################
    dnn = DNN(sizes, seed=seed)
    dnn.set_inference_params(layerParams,
        memory_window=memory_window,
        batchnorm_style=batchnorm_style,
        fold_batchnorm=fold_batchnorm)
    dnn.init_GPU(useGPU,gpu_id)
    
    Nlayers = len(paramsList)
    Nlayers_mvm = 0
    for k in range(Nlayers):
        if layerParams[k]['activation'] is None:
            dnn.set_activations(layer=k,style="NONE")
        elif layerParams[k]['activation']['type'] == "RECTLINEAR":
            dnn.set_activations(layer=k,style=layerParams[k]['activation']['type'],relu_bound=layerParams[k]['activation']['bound'])
        elif layerParams[k]['activation']['type'] == "QUANTIZED_RELU":
            dnn.set_activations(layer=k,style=layerParams[k]['activation']['type'],nbits=layerParams[k]['activation']['nbits'])
        elif layerParams[k]['activation']['type'] == "WHETSTONE":
            dnn.set_activations(layer=k,style=layerParams[k]['activation']['type'],sharpness=layerParams[k]['activation']['sharpness'])
        elif layerParams[k]['activation']['type'] == "CLIP":
            dnn.set_activations(layer=k,style=layerParams[k]['activation']['type'],clip_min=layerParams[k]['activation']['min'],clip_max=layerParams[k]['activation']['max'])
        else:
            dnn.set_activations(layer=k,style=layerParams[k]['activation']['type'])

    print("Initializing neural cores")
    # weight_dict_ = dict([(layer.name, layer.get_weights()) for layer in keras_model.layers])
    # weight_list_=[]
    # for jj in weight_dict_.keys():
    #     if len(weight_dict_[jj]) > 0 :
    #         weight_list_.append(weight_dict_[jj][0])

    
    time_start = time.time()
    weight_dict = {}
    params = {}
    for tensor in onnx_model.graph.initializer:
        params[tensor.name] = numpy_helper.to_array(tensor)
    

    if quantize:
        p=0
        for node in onnx_model.graph.node:
            # 노드별로 가중치와 편향을 저장할 변수 초기화
            w = None
            b = None
            w_ =None
            # 노드의 입력 중 'W'와 'B'에 해당하는 부분 찾기
            # print('p',p)
            for idx, input_name in enumerate(node.input):
                if input_name in weight_for_quantized:
                    tensor = weight_for_quantized[input_name]
                    if 'weight' in input_name or idx == 1:
                        if node.op_type == 'Conv' :
                            w = tensor
                            # w_ = weight_list_[p].transpose(3,2,0,1)
                            p=p+1
                        else:
                            w = tensor
                            # w_ = weight_list_[p].transpose()
                            
                            # w=tensor
                            # w = weight_list_[p]
                            p=p+1
                    elif 'bias' in input_name or idx == 2:
                        b = tensor
            if w is not None :
                weight_dict[node.name] = [w, b]

    else:
        p=0
        for node in onnx_model.graph.node:
            # 노드별로 가중치와 편향을 저장할 변수 초기화
            w = None
            b = None
            w_ =None
            # 노드의 입력 중 'W'와 'B'에 해당하는 부분 찾기
            # print('p',p)
            for idx, input_name in enumerate(node.input):
                if input_name in params:
                    tensor = params[input_name]
                    if 'weight' in input_name or idx == 1:
                        if node.op_type == 'Conv' :
                            w = tensor
                            # w_ = weight_list_[p].transpose(3,2,0,1)
                            p=p+1
                        else:
                            w = tensor
                            # w_ = weight_list_[p].transpose()
                            
                            # w=tensor
                            # w = weight_list_[p]
                            p=p+1
                    elif 'bias' in input_name or idx == 2:
                        b = tensor
            if w is not None :
                weight_dict[node.name] = [w, b]

    # for jj in weight_dict.keys():
    #     print(jj)
    #     print(weight_dict[jj])
    
    layerNums =[]
    ncoresList =[]
    initialization = True
    print(time.time()-time_start)
    time_start = time.time()
   
    for m in range(Nlayers):
        params_m = paramsList[m]
        if layerParams[m]['type'] not in ("conv","dense"):
            dnn.set_layer_params(m, layerParams[m], digital_bias)
            continue
        


        Nlayers_mvm += 1

        Ncores = (len(params_m) if type(params_m) is list else 1)
        if (layerParams[m]['type'] in ("conv") and layerParams[m]['depthwise']==False) or layerParams[m]['type'] in ("dense"):
            layerNums.append(m)
            ncoresList.append(Ncores)
        # print()
        Wm_0 = weight_dict[layerParams[m]['name']]
        Wm = Wm_0[0]
        if layerParams[m]['bias']:
            Wbias = Wm_0[1]

        # Quantize weights here if model is Larq
        if layerParams[m]['binarizeWeights']:
            Wm = np.sign(Wm)

        if fold_batchnorm and layerParams[m]['batch_norm'] is not None:
            if layerParams[m]['BN_scale'] and layerParams[m]['BN_center']:
                gamma, beta, mu, var = weight_dict[layerParams[m]['batch_norm']]
            elif layerParams[m]['BN_scale'] and not layerParams[m]['BN_center']:
                gamma, mu, var = weight_dict[layerParams[m]['batch_norm']]
                beta = 0
            elif not layerParams[m]['BN_scale'] and layerParams[m]['BN_center']:
                beta, mu, var = weight_dict[layerParams[m]['batch_norm']]
                gamma = 1
            else:
                mu, var = weight_dict[layerParams[m]['batch_norm']]
                gamma, beta = 1, 0
            
            epsilon = layerParams[m]['epsilon']
            if not layerParams[m]['bias']:
                Wbias = np.zeros(Wm.shape[-1])
                layerParams[m]['bias'] = True

            if batchnorm_style == "sqrt":
                if not (layerParams[m]['type'] == 'conv' and layerParams[m]['depthwise']):
                    Wm = gamma*Wm/np.sqrt(var + epsilon)
                    Wbias = (gamma/np.sqrt(var + epsilon))*(Wbias-mu) + beta
                else:
                    Wm = gamma[None,None,:,None]*Wm/np.sqrt(var[None,None,:,None] + epsilon)
                    Wbias = (gamma[None,None,:,None]/np.sqrt(var[None,None,:,None] + epsilon))*(Wbias-mu[None,None,:,None]) + beta[None,None,:,None]
                    Wbias = np.squeeze(Wbias)

            elif batchnorm_style == "no_sqrt":
                if not (layerParams[m]['type'] == 'conv' and layerParams[m]['depthwise']):
                    Wm = gamma*Wm/(var + epsilon)
                    Wbias = (gamma/(var + epsilon))*(Wbias-mu) + beta
                else:
                    Wm = gamma[None,None,:,None]*Wm/(var[None,None,:,None] + epsilon)
                    Wbias = (gamma[None,None,:,None]/(var[None,None,:,None] + epsilon))*(Wbias-mu[None,None,:,None]) + beta[None,None,:,None]
                    Wbias = np.squeeze(Wbias)

            weight_dict[layerParams[m]['name']] = [Wm,Wbias]

        ### Explicit handling of quantized neural networks
        if model_name == "Resnet50" or model_name == "MobileNetV2" or model_name == "Resnet50-int4":
            params_m = qnn_adjustment(model_name, params_m.copy(), Wm, Ncores, Nlayers_mvm)
        # set neural core parameters
        if digital_bias:
            layerParams[m]['bias_row'] = False
        else:
            layerParams[m]['bias_row'] = layerParams[m]['bias']

        if layerParams[m]['type'] == "conv":
            if Ncores == 1:
                params_m.simulation.convolution.bias_row = layerParams[m]['bias_row']
            else:
                for k in range(Ncores):
                    params_m[k].simulation.convolution.bias_row = layerParams[m]['bias_row']

        # Set # images to profile
        if Ncores == 1 and params_m.simulation.analytics.profile_adc_inputs:
            if dataset == "imagenet" and ntest > 50 and params_m.xbar.dac.mvm.input_bitslicing:
                print("Warning: Using >50 ImageNet images in bitwise BL current profiling. Might run out of memory!")
            params_m.simulation.analytics.ntest = ntest
        elif Ncores > 1 and params_m[0].simulation.analytics.profile_adc_inputs:
            if dataset == "imagenet" and ntest > 50 and params_m[0].xbar.dac.mvm.input_bitslicing:
                print("Warning: Using >50 ImageNet images in bitwise BL current profiling. Might run out of memory!")
            for k in range(Ncores):
                params_m[k].simulation.analytics.ntest = ntest

        dnn.set_layer_params(m, layerParams[m], digital_bias)
        dnn.ncore(m, style=layerParams[m]['type'], params=params_m)

    # Import weights to CrossSim cores
    print(time.time()-time_start)
    time_start = time.time()
    dnn.read_weights_keras(weight_dict)
    print(time.time()-time_start)
    time_start = time.time()
    dnn.expand_cores()
    print(time.time()-time_start)
    # Import bias weights to be added digitally
    if digital_bias:
        dnn.import_digital_bias(weight_dict,bias_bits)

    # If using int4 model, import quantization and scale factors
    if model_name == "Resnet50-int4":
        dnn.import_quantization(weight_dict)
        dnn.import_scale(weight_dict)

    # Get a params object to use for easy access to simulation params
    
    params_0 = (paramsList[fisrt_conv_dense][0] if type(paramsList[fisrt_conv_dense])==list else paramsList[fisrt_conv_dense])

    # if onnx_model is not None and search_accuracy is False:
    #     del Wm, weight_dict
    #     K.clear_session()
    # else:
    #     del Wm
    print(time.time()-time_start)
    if randomSampling:
        ntest_batch = ntest
        print("Warning: ntest_batch is ignored with random sampling")

    if show_HW_config:
        print("\n\n============================")
        print("Analog HW configuration")
        print("============================\n")
        dnn.show_HW_config()
        input("\nPress any key to continue...")

    if ntest_batch > 0:
        if ntest_batch > ntest:
            ntest_batch = ntest
        nloads = (ntest-1) // ntest_batch + 1
        if type(topk) is int:
            frac_accum = 0
        else:
            frac_accum = np.zeros(len(topk))        
    else:
        print('Loading dataset')
        ntest_batch = ntest
        nloads = 1
    nstart_i = nstart
    network_outputs = None

    for nl in range(nloads):

        ## Load truncated data set
        nend_i = nstart_i + ntest_batch
        if nloads > 1:
            print('Loading dataset, images {:d} to {:d} of {:d}'.format(nl*ntest_batch,(nl+1)*ntest_batch,ntest))

        # Set FP precision based on the ADC/DAC precision: use 24 bits as cut-off
        if params_0.xbar.adc.mvm.bits > 24 or params_0.xbar.dac.mvm.bits > 24:
            precision = np.float64
        else:
            precision = np.float32

        # Load input data and labels
        (x_test, y_test) = load_dataset_inference(dataset, 
                                nstart_i, nend_i, 
                                calibration = calibration,
                                precision = precision,
                                subtract_pixel_mean = subtract_pixel_mean,
                                imagenet_preprocess = imagenet_preprocess,
                                dataset_normalization = dataset_normalization)

        # If first layer is convolution, transpose the dataset so channel comes first

        # if params_0.simulation.convolution.is_conv_core:
            # x_test = np.transpose(x_test,(0,3,1,2))

        dnn.indata = x_test
        dnn.answers = y_test
        dnn.ndata = (ntest_batch if not randomSampling else x_test.shape[0])

        # If the first layer is using GPU, send the inputs to the GPU
        if useGPU:
            import cupy as cp
            cp.cuda.Device(gpu_id).use()
            dnn.indata = cp.array(dnn.indata)

        # Run inference
        if search_accuracy:
            break
        print("Beginning inference. Truncated test set to %d examples" % ntest)
        time_start = time.time()
        count, frac, network_output = dnn.predict(
                n = ntest_batch,
                count_interval = count_interval,
                time_interval = time_interval,
                randomSampling = randomSampling,
                topk = topk,
                return_network_output = return_network_output,
                profiling_folder = profiling_folder,
                profiling_settings = profiling_settings,
                weight_dict = weight_dict)

        # Print accuracy results
        time_stop = time.time()
        device = "CPU" if not useGPU else "GPU"
        sim_time = time_stop - time_start
        print("Total " + device + " seconds = {:.3f}".format(sim_time))
        if type(topk) is int:
            print("Inference accuracy: {:.3f}% ({:d}/{:d})".format(frac*100,count,ntest_batch))
        else:
            accs = ""
            for j in range(len(topk)):
                accs += "{:.2f}% (top-{:d}, {:d}/{:d})".format(100*frac[j], topk[j], count[j], ntest_batch)
                if j < (len(topk)-1): accs += ", "
            print("Inference acuracy: "+accs+"\n")

        # Consolidate neural network outputs
        if return_network_output:
            if nloads == 1:
                network_outputs = network_output
            else:
                if nl == 0:
                    network_outputs = np.zeros((ntest, network_output.shape[1]))
                network_outputs[(nl*ntest_batch):((nl+1)*ntest_batch),:] = network_output

        if nloads > 1:
            nstart_i += ntest_batch
            frac_accum += frac
    if search_accuracy is False:
        return frac, network_outputs , layerNums, ncoresList
    
    loop_for_search = True
    if search_accuracy:
        if config.searchADC:
            len_adc_candidate = len(adc_ranges_cadidate[0][0])
        else:
            len_adc_candidate = 1
        mvm = 0
        mvm_wo_depthwise = 0
        feature_of_ort = None
        storage_from_n = None

        best_mse_idx =[]
        mse_list = []
    

    print(len_adc_candidate)

    beam_list_feature_map  = [[None,None,None]]
    beam_width = 1
    beam_list_index = [[] for _ in range(beam_width)]
    adc_list_index = [[] for _ in range(beam_width)]
    


    while loop_for_search and search_accuracy:
        if mvm == config.Nlayers_mvm:
            break
        print("----------------------------------", mvm, "----------------------------------")
        print(":::", mvm_wo_depthwise, ":::")
        beam_tmp=[]
        for beam_idx in range(len(beam_list_feature_map)):
            storage_tmp = [[0 for _ in range(len_adc_candidate)] for _ in range(len(index_row_col))]
            storage_feature_of_candidate = [[[None for _ in range(3)]for _ in range(len_adc_candidate)] for _ in range(len(index_row_col))]
            storage_mse = [[None for _ in range(len_adc_candidate)] for _ in range(len(index_row_col))]
            for row_col_cadidate_idx in range(len(index_row_col)): #-1, -1, -1
                for adc_cadidate_idx in range(len_adc_candidate):

                    if validate_mse:
                        if row_col_cadidate_idx != validate_seq[mvm]:
                            continue
                    
                    print("row_col_cadidate_idx", row_col_cadidate_idx)
                    print("adc_cadidate_idx", adc_cadidate_idx)
                    m = mvm_to_n[mvm]
                    

                    if layerParams[m]['type'] not in ("conv","dense"):
                        continue
                    if layerParams[m]['type'] in ("conv") and layerParams[m]['depthwise']:
                        continue
                    #----------------------------------set_params----------------------------------
                    if layerParams[m]['type'] == 'conv':
                        Nrows = layerParams[m]['Kx']*layerParams[m]['Ky']*layerParams[m]['Nic']
                        Ncols = layerParams[m]['Noc']
                        if not config.digital_bias:
                            if layerParams[m]['bias'] or config.fold_batchnorm:
                                Nrows += 1

                    elif layerParams[m]['type'] == 'dense':
                        Nrows = sizes[m][2]
                        Ncols = sizes[m+1][2]
                        if not config.digital_bias and layerParams[m]['bias']:
                            Nrows += 1

                    # Compute number of arrays matrix must be partitioned across

                    NrowsMax = index_row_col[row_col_cadidate_idx][0]
                    NcolsMax = index_row_col[row_col_cadidate_idx][1]
                    adc_range = adc_ranges_cadidate[row_col_cadidate_idx][mvm_wo_depthwise][adc_cadidate_idx]
                    if adc_range is not None:
                        adc_range = adc_range.tolist()

                    print("adc_range", adc_range)
                    if NrowsMax > 0:
                        Ncores = (Nrows-1)//NrowsMax + 1
                    else:
                        Ncores = 1

                    
                    if NcolsMax > 0:
                        Ncores *= (Ncols-1)//NcolsMax + 1
                    else:
                        Ncores *= 1

                    # Layer specific ADC and activation resolution and range (set above)
                    
                    try:
                        paramsList[m][0]
                        tmp_param = paramsList[m][0].copy()      
                    except:
                        tmp_param = paramsList[m].copy()      

                    dac_range = tmp_param.dac_range
                    adc_bits_j = tmp_param.xbar.adc.mvm.bits
                    dac_bits_j = tmp_param.xbar.dac.mvm.bits
                    
                    x_par = tmp_param.simulation.convolution.x_par
                    y_par = tmp_param.simulation.convolution.y_par
                    convParams = tmp_param.convParams

                    Icol_max_norm = tmp_param.xbar.array.Icol_max 
                    positiveInputsOnly = tmp_param.positiveInputsOnly
                    # Does this layer use analog batchnorm?
                    analog_batchnorm = config.fold_batchnorm and layerParams[m]['batch_norm'] is not None

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
                        positiveInputsOnly=positiveInputsOnly,
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
                        paramsList[m] = params
                    else:
                        paramsList[m] = Ncores*[None]
                        for k in range(Ncores):
                            paramsList[m][k] = params.copy()            


                    
                    params_m = paramsList[m]
                    
            
                    Ncores = (len(params_m) if type(params_m) is list else 1)
                
                    Wm_0 = weight_dict[layerParams[m]['name']]
                    Wm = Wm_0[0]
                    if layerParams[m]['bias']:
                        Wbias = Wm_0[1]

                    # Quantize weights here if model is Larq
                    if layerParams[m]['binarizeWeights']:
                        Wm = np.sign(Wm)

                    if fold_batchnorm and layerParams[m]['batch_norm'] is not None:
                        raise Exception("Batchnorm folding not supported for accuravcy search")

                        
                    ### Explicit handling of quantized neural networks
                    if model_name == "Resnet50" or model_name == "MobileNetV2" or model_name == "Resnet50-int4":
                        params_m = qnn_adjustment(model_name, params_m.copy(), Wm, Ncores, mvm)

                    # set neural core parameters
                    if digital_bias:
                        layerParams[m]['bias_row'] = False
                    else:
                        layerParams[m]['bias_row'] = layerParams[m]['bias']

                    if layerParams[m]['type'] == "conv":
                        if Ncores == 1:
                            params_m.simulation.convolution.bias_row = layerParams[m]['bias_row']
                        else:
                            for k in range(Ncores):
                                params_m[k].simulation.convolution.bias_row = layerParams[m]['bias_row']

                    # Set # images to profile
                    if Ncores == 1 and params_m.simulation.analytics.profile_adc_inputs:
                        if dataset == "imagenet" and ntest > 50 and params_m.xbar.dac.mvm.input_bitslicing:
                            print("Warning: Using >50 ImageNet images in bitwise BL current profiling. Might run out of memory!")
                        params_m.simulation.analytics.ntest = ntest
                    elif Ncores > 1 and params_m[0].simulation.analytics.profile_adc_inputs:
                        if dataset == "imagenet" and ntest > 50 and params_m[0].xbar.dac.mvm.input_bitslicing:
                            print("Warning: Using >50 ImageNet images in bitwise BL current profiling. Might run out of memory!")
                        for k in range(Ncores):
                            params_m[k].simulation.analytics.ntest = ntest
                    dnn.set_layer_params(m, layerParams[m], digital_bias)
                    dnn.ncore(m, style=layerParams[m]['type'], params=params_m)
                
                    dnn.read_weights_keras_one_layer(weight_dict,m)
                    dnn.expand_cores()


                    if digital_bias:
                        dnn.import_digital_bias(weight_dict,bias_bits)

                    # If using int4 model, import quantization and scale factors
                    if model_name == "Resnet50-int4":
                        dnn.import_quantization(weight_dict)
                        dnn.import_scale(weight_dict)

                    # Get a params object to use for easy access to simulation params
                    params_0 = (paramsList[fisrt_conv_dense][0] if type(paramsList[fisrt_conv_dense])==list else paramsList[fisrt_conv_dense])

                    if randomSampling:
                        ntest_batch = ntest
                        print("Warning: ntest_batch is ignored with random sampling")

                    if show_HW_config:
                        print("\n\n============================")
                        print("Analog HW configuration")
                        print("============================\n")
                        dnn.show_HW_config()
                        input("\nPress any key to continue...")

                    if ntest_batch > 0:
                        if ntest_batch > ntest:
                            ntest_batch = ntest
                        nloads = (ntest-1) // ntest_batch + 1
                        if type(topk) is int:
                            frac_accum = 0
                        else:
                            frac_accum = np.zeros(len(topk))        
                    else:
                        print('Loading dataset')
                        ntest_batch = ntest
                        nloads = 1
                    nstart_i = nstart
                    network_outputs = None

                    for nl in range(nloads):

                        # Run inference
                        print("Beginning inference. Truncated test set to %d examples" % ntest)
                        storage_mse[row_col_cadidate_idx][adc_cadidate_idx], feature_of_ort, storage_feature_of_candidate[row_col_cadidate_idx][adc_cadidate_idx][0], storage_feature_of_candidate[row_col_cadidate_idx][adc_cadidate_idx][1], storage_feature_of_candidate[row_col_cadidate_idx][adc_cadidate_idx][2], new_from_n = dnn.predict(
                                n = ntest_batch,
                                count_interval = count_interval,
                                time_interval = time_interval,
                                randomSampling = randomSampling,
                                topk = topk,
                                return_network_output = return_network_output,
                                profiling_folder = profiling_folder,
                                profiling_settings = profiling_settings,
                                weight_dict = weight_dict,
                                search_accuracy=search_accuracy,
                                search_accuracy_layer=mvm_wo_depthwise,
                                config=config,
                                ort_output_list =feature_of_ort,
                                search_accuracy_output_vec_list = beam_list_feature_map[beam_idx][0],
                                search_accuracy_output_vecs_list = beam_list_feature_map[beam_idx][1],
                                search_accuracy_output_vecs_add_list = beam_list_feature_map[beam_idx][2],
                                from_n = storage_from_n,
                                metric=metric,
                        )
                        
                        storage_tmp[row_col_cadidate_idx][adc_cadidate_idx] = [storage_mse[row_col_cadidate_idx][adc_cadidate_idx],storage_feature_of_candidate[row_col_cadidate_idx][adc_cadidate_idx]]
                        print("storage_mse", storage_mse[row_col_cadidate_idx][adc_cadidate_idx])
            # print(storage_feature_of_candidate[0][0][0][0])
            # print(storage_feature_of_candidate[1][0][0][0])

            beam_tmp.append(storage_tmp.copy())
            # Print accuracy results
        if layerParams[m]['type'] in ("dense") or layerParams[m]['type'] in ("conv") and layerParams[m]['depthwise']==False:
            index_value_pairs = []
            for beam_tmp_idx in range(len(beam_tmp)):
                for row_col_cadidate_idx in range(len(index_row_col)):
                    for adc_cadidate_idx in range(len_adc_candidate):
                        if validate_mse:
                            if row_col_cadidate_idx != validate_seq[mvm]:
                                continue
                        value = beam_tmp[beam_tmp_idx][row_col_cadidate_idx][adc_cadidate_idx][0].copy()
                        feature = beam_tmp[beam_tmp_idx][row_col_cadidate_idx][adc_cadidate_idx][1].copy()
                        index_value_pairs.append(((beam_tmp_idx, row_col_cadidate_idx, adc_cadidate_idx),feature, value))
                        print("row_col_cadidate_idx", row_col_cadidate_idx, value)

            sorted_pairs = sorted(index_value_pairs, key=lambda x: (x[2], -x[0][1]))            
            sorted_index = [x[0] for x in sorted_pairs]
            sorted_value = [x[2] for x in sorted_pairs]
            

            check_same_index = {}
            check_same_index_value = {}
            for s_idx , s in enumerate(sorted_index):
                if str(beam_list_index[s[0]])+str(s[2]) in check_same_index.keys() and check_same_index_value[str(beam_list_index[s[0]])+str(s[2])] == sorted_value[s_idx]:
                    check_same_index[str(beam_list_index[s[0]])+str(s[2])] = True
                    continue
                elif str(beam_list_index[s[0]])+str(s[2]) in check_same_index.keys() and check_same_index_value[str(beam_list_index[s[0]])+str(s[2])] != sorted_value[s_idx]:
                    check_same_index[str(beam_list_index[s[0]])+str(s[2])] = False
                    break
                check_same_index[str(beam_list_index[s[0]])+str(s[2])] = False
                check_same_index_value[str(beam_list_index[s[0]])+str(s[2])] = sorted_value[s_idx]

                
            b_w = 0 
            beam_list_feature_map = []
            beam_list_index_next = []
            adc_list_index_next = []
            already_append = []
            for s_idx , s in enumerate(sorted_index):
                if str(beam_list_index[s[0]])+str(s[2]) not in already_append:
                    if check_same_index[str(beam_list_index[s[0]])+str(s[2])]==True:
                        already_append.append(str(beam_list_index[s[0]])+str(s[2]))
                        prev = beam_list_index[s[0]].copy()
                        prev.append(-1)
                        beam_list_index_next.append(prev)

                        prev_adc = adc_list_index[s[0]].copy()
                        prev_adc.append(s[2])
                        adc_list_index_next.append(prev_adc)

                        beam_list_feature_map.append(sorted_pairs[s_idx][1].copy())
                    else:
                        prev = beam_list_index[s[0]].copy()
                        prev.append(s[1])
                        beam_list_index_next.append(prev)

                        prev_adc = adc_list_index[s[0]].copy()
                        prev_adc.append(s[2])
                        adc_list_index_next.append(prev_adc)

                        beam_list_feature_map.append(sorted_pairs[s_idx][1].copy())
                    b_w += 1
                    if b_w == beam_width:
                        break
                
            
                # beam_list_feature_map = [x[1].copy() for x in sorted_pairs[:beam_width]]
            beam_list_index = beam_list_index_next
            adc_list_index = adc_list_index_next
            storage_from_n = new_from_n
            mvm_wo_depthwise += 1
            
        elif (layerParams[m]['type'] in ("conv") and layerParams[m]['depthwise']==True):
            for i in beam_list_index:
                i.append(None)
            for i in adc_list_index:
                i.append(None)


        print("storage_mse", storage_mse)
        print("sorted_value", sorted_value[:beam_width])
        print("beam_list_index", beam_list_index)
        print("adc_list_index", adc_list_index)
        mse_list.append(storage_mse)
        # print('mse_list', mse_list)
        mvm += 1
        # #compare mse
        # if layerParams[m]['type'] in ("dense") or layerParams[m]['type'] in ("conv"):# and layerParams[m]['depthwise']==False):
        #     best = None
        #     best_row_col_cadidate_idx = None
        #     best_adc_cadidate_idx = None
        #     same = True
        #     for row_col_cadidate_idx in range(len(index_row_col)):
        #         for adc_cadidate_idx in range(len_adc_candidate):
        #             if best is not None and storage_mse[row_col_cadidate_idx][adc_cadidate_idx] != best:
        #                 same = False

        #             if best is None or storage_mse[row_col_cadidate_idx][adc_cadidate_idx] < best:
        #                 best = storage_mse[row_col_cadidate_idx][adc_cadidate_idx]
        #                 best_row_col_cadidate_idx = row_col_cadidate_idx
        #                 best_adc_cadidate_idx = adc_cadidate_idx
                    

        #     for row_col_cadidate_idx in range(len(index_row_col)):
        #         for adc_cadidate_idx in range(len_adc_candidate):
        #             if not(row_col_cadidate_idx == best_row_col_cadidate_idx and adc_cadidate_idx == best_adc_cadidate_idx):
        #                 storage_feature_of_candidate[row_col_cadidate_idx][adc_cadidate_idx][0] = storage_feature_of_candidate[best_row_col_cadidate_idx][best_adc_cadidate_idx][0].copy()
        #                 storage_feature_of_candidate[row_col_cadidate_idx][adc_cadidate_idx][1] = storage_feature_of_candidate[best_row_col_cadidate_idx][best_adc_cadidate_idx][1].copy()
        #                 storage_feature_of_candidate[row_col_cadidate_idx][adc_cadidate_idx][2] = storage_feature_of_candidate[best_row_col_cadidate_idx][best_adc_cadidate_idx][2].copy()

        #     print("storage_mse", storage_mse)
        #     print("best_row_col_cadidate_idx", best_row_col_cadidate_idx)
        #     print("best_adc_cadidate_idx", best_adc_cadidate_idx)

        #     if same:
        #         # best_mse_idx.append([-1,-1])
        #         best_mse_idx.append(-1)
        #     else:
        #         # best_mse_idx.append([best_row_col_cadidate_idx,best_adc_cadidate_idx])
        #         best_mse_idx.append(best_row_col_cadidate_idx)
        #     a = []
        #     for i in range(len(storage_mse)):
        #         t = []
        #         for j in range(len(storage_mse[i])):
        #             t.append(storage_mse[i][j])
        #         a.append(t)    
                    
        #     mse_list.append(a)
        #     print('mse_list', mse_list)
        #     print('best_mse_idx', best_mse_idx)
        #     storage_from_n = new_from_n
        
        # loop_for_search = False
    best_mse_idx =[beam_list_index[0],adc_list_index[0]]
    return best_mse_idx, mse_list