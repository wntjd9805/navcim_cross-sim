import onnx
from onnx import numpy_helper
import os
import tarfile
import numpy as np
# ONNX 모델 불러오기

# os.system("git lfs pull /root/models/vision/classification/vgg/model/vgg16-12-int8.onnx")



# 모델에서 각 노드의 가중치 추출
def extract_bias_values(model):
    graph = model.graph

    weights = {}
    for tensor in graph.initializer:
        numpy_array = numpy_helper.to_array(tensor)
        weights[tensor.name] = numpy_array

    # 바이어스 텐서의 이름을 찾고 값을 출력
    # print(weights)
    # for node in graph.node:
        # print(node.op_type)
        # if node.op_type == 'QLinearConv':
            # print(node.input)
            # print(any('bias' in name for name in node.input))
           
# extract_bias_values(model)
def add_constants_to_dict(weights, model):
    # 모델을 로드합니다.
    graph = model.graph

    # 그래프의 모든 노드를 순회하며 상수를 찾고 weights 딕셔너리에 추가합니다.
    for node in graph.node:
        if node.op_type == "Constant":
            # 상수 노드의 속성을 추출합니다.
            for attribute in node.attribute:
                if attribute.name == "value":
                    # 텐서 값을 numpy 배열로 변환하고 weights 딕셔너리에 추가합니다.
                    tensor_value = numpy_helper.to_array(attribute.t)
                    weights[node.output[0]] = tensor_value

    return weights


def isActivation(op_type):
    return op_type in ("Relu", "LeakyRelu", "Sigmoid", "Softmax","Clip")

def isMainLayerType(op_type):
    return op_type in ('Conv', 'MaxPool', 'AveragePool', 'GlobalAveragePool', 'GlobalMaxPool', 'Gemm', 'Add','Mul' ,'Concat', 'QuantizeLinear', 'DequantizeLinear', 'Flatten','QLinearConv','QLinearMatMul','QLinearAdd',"ReduceMean")

def searchForLayer(layerName,layerParams,exist_wo_activate = False):
    
    for j in range(len(layerParams)):
        if (layerParams[j]['name'] == layerName) \
        or (layerParams[j]['batch_norm'] == layerName) or (layerParams[j]['appended'] == layerName) \
        or (layerParams[j]['activation'] is not None and layerParams[j]['activation']['name'] == layerName):
            return j
    raise ValueError(f"Source layer could not be found {layerName}")


ignoredLayerTypes = ["Dropout","GaussianNoise","Shape","Cast","ConstantOfShape"]
def get_onnx_input_info(model, task="imagenet"):
    # 모델 로드
    graph = model.graph

    # 첫 번째 입력 노드 (일반적으로 모델의 입력을 나타냄)
    if len(graph.input) > 0:
        first_input = graph.input[0]
        input_name = first_input.name

        # 입력 노드의 형태 정보 추출
        input_shape = [dim.dim_value for dim in first_input.type.tensor_type.shape.dim]

        # 입력 차원 확인 및 처리
        if len(input_shape) == 4: # 일반적으로 [배치 크기, 채널, 높이, 너비]
            Nix0 = input_shape[2] # 높이
            Niy0 = input_shape[3] # 너비
            Nic0 = input_shape[1] # 채널

            # 입력 차원이 정의되지 않은 경우
            if Nix0 is None or Niy0 is None:
                print('Input dimensions not defined. Using ' + task + ' image size')
                if task in ("cifar10", "cifar100"):
                    Nix0, Niy0 = 32, 32
                elif task in ("mnist", "fashion"):
                    Nix0, Niy0 = 28, 28
                elif task == "tinyimagenet":
                    Nix0, Niy0 = 64, 64
                elif task == "imagenet":
                    Nix0, Niy0 = 224, 224
                else:
                    raise ValueError("Image size not specified and dataset unknown")
        else:
            raise ValueError("Unsupported input shape")

        return input_name, Nix0, Niy0, Nic0
    else:
        raise ValueError("No input node found in the model")
    


def get_onnx_metadata(model,model_name,quantized,debug_graph=False, search_accuracy =False):
    print('Reading ONNX model metadata...')
    # ONNX 모델 로드
    graph = model.graph
    # ONNX 모델의 노드 및 텐서 정보를 추출하기 위한 기본 구조
    nodes = graph.node
    tensors = graph.initializer
    weights_ = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}
    weights = add_constants_to_dict(weights_, model)

    model_info = get_onnx_input_info(model)
    print(model_info)
    inputlayer_name,Nix0, Niy0, Nic0 = get_onnx_input_info(model)
    layerParams = []
    # print(inputlayer_name,Nix0, Niy0, Nic0 )
    layerParams_first = {}
    layerParams_first['name'] = inputlayer_name
    layerParams_first['Nox'] = Nix0
    layerParams_first['Noy'] = Niy0
    layerParams_first['Noc'] = Nic0
    layerParams_first['Nix'] = Nix0
    layerParams_first['Niy'] = Niy0
    layerParams_first['Nic'] = Nic0
    layerParams_first['type'] = 'input'
    layerParams_first['source'] = None
    layerParams_first['activation'] = None


    # ONNX 모델의 각 노드 및 연산 분석
    node_input_output = {}
    weight_for_quantized={}

    output_to_next_node = {output: next_node for node in graph.node for output in node.output for next_node in graph.node if next_node.input and next_node.input[0] == output}
    for node in nodes:
        # 노드 이름, 유형 등 기본 정보
        node_name = node.name
        # print(node_name)
        class_name = node.op_type
        # print('node_name:',node_name)
        # print('class_name: ', class_name)
        # 입력 및 출력 텐서 추출
        input_tensors = node.input
        output_tensors = node.output
        
        # Check if layer is a main layer type
        if isMainLayerType(class_name):
            node_input_output[node.output[0]] = node.name
            layerParams_k = {}
            layerParams_k['name'] = node.name
            layerParams_k['output_name'] = node.output[0]
            # Default settings: may change below
            layerParams_k['batch_norm'] = None
            layerParams_k['appended'] = None
            layerParams_k['bias'] = False
            layerParams_k['activation'] = None
            layerParams_k['splitBeforeBN'] = False

            ###################
            ##  CONVOLUTION  ##
            ###################
            # print(class_name)
            if class_name in ('Conv', 'QLinearConv'):
                layerParams_k['type'] = 'conv'
                layerParams_k['bias']  = len(node.input) == 3 
                # layerParams_k['bias'] = any('bias' in name for name in node.input)
                layerParams_k['binarizeWeights'] = False # used only for Larq, changed below if detected
                layerParams_k['sameConv']=True
                layerParams_k['px_0'], layerParams_k['px_1'] = 0, 0
                layerParams_k['py_0'], layerParams_k['py_1'] = 0, 0
                for attr in node.attribute:
                    if attr.name == 'strides':
                        layerParams_k['stride'] = attr.ints[0]  # 가정: 모든 차원에 대해 동일한 stride
                    elif attr.name == 'kernel_shape':
                        layerParams_k['Kx'], layerParams_k['Ky'] = attr.ints
                    elif attr.name == 'pads':
                        # 패딩 정보 처리. ONNX는 시작 및 끝 패딩을 구별하여 제공할 수 있음
                        layerParams_k['px_0'], layerParams_k['py_0'], layerParams_k['px_1'], layerParams_k['py_1'] = attr.ints
                        layerParams_k['sameConv'] = False
                        # layerParams_k['padding'] = 'same'
                    elif attr.name == 'group':
                        group = attr.i
                        if group != 1:
                            layerParams_k['depthwise'] = True
                            layerParams_k['group'] = group
                        else:
                            layerParams_k['depthwise'] = False
                # Check if convolution is depthwise and set # output channels (set later for depthwise)
                if len(layerParams) == 0: # Get dimensions from InputLayer
                    layerParams_k['Nix'] = Nix0
                    layerParams_k['Niy'] = Niy0
                    layerParams_k['Nic'] = Nic0

                if quantized == False :
                    if not layerParams_k['depthwise']:
                            # filter = [name for name in node.input if 'weight_quantized' in name]
                            # layerParams_k['Noc'] = weights[filter[0]].shape[0]
                        # else:
                            weight_name = node.input[1]
                            layerParams_k['Noc'] = weights[weight_name].shape[0]

                    # Input shape can be specified for the first conv layer, computed later for the other layers
                    # Find source layer, if no inbound nodes set, assume sequential
                    if len(layerParams) > 0:
                        input1 = node_input_output.get(node.input[0], None)
                        if input1 is not None: 
                            k_src = searchForLayer(input1,layerParams)
                        else:                        
                            k_src = len(layerParams)-1
                        layerParams_k['source'] = np.array([k_src])
                        layerParams_k['splitBeforeBN'] = ((k_src in layerParams) and (((layerParams[k_src]['type'] == 'add') and ('add' in input1)) or ((layerParams[k_src]['type'] == 'mul') and ('mul' in input1))))
                    else:
                        k_src = len(layerParams)-1
                        layerParams_k['source'] = np.array([k_src]) # First layer
                else :
                    input_q = node_input_output.get(node.input[0], None)
                    input_0 = searchForLayer(input_q, layerParams)
                    layerParams_k['x_scale'] = layerParams[input_0]['scale'] 
                    layerParams_k['x_zero_point'] = layerParams[input_0]['zero_point']
                    layerParams[input_0]['needless'] = True

                    input_q_1 = node_input_output.get(node.input[1], None)
                    input_1 = searchForLayer(input_q_1, layerParams)
                    layerParams_k['Noc'] = layerParams[input_1]['Noc'] 
                    layerParams_k['w_scale'] = layerParams[input_1]['scale'] 
                    layerParams_k['w_zero_point'] = layerParams[input_1]['zero_point']
                    layerParams[input_1]['needless'] = True

        
                    if len(layerParams) > 0:
                        input1 = node_input_output.get(node.input[0], None)
                        if input1 is not None: 
                            k_src = searchForLayer(input1,layerParams)
                        else:                        
                            k_src = len(layerParams)-1
                        layerParams_k['source'] = np.array([k_src])
                        layerParams_k['splitBeforeBN'] = ((k_src in layerParams) and (((layerParams[k_src]['type'] == 'add') and ('add' in input1)) or ((layerParams[k_src]['type'] == 'mul') and ('mul' in input1))))
                    else:
                        k_src = len(layerParams)-1
                        layerParams_k['source'] = np.array([k_src]) # First layer

                    # Nsources = len(node.input)
                    # k_srcs = np.zeros(Nsources, dtype=int)
                    # for q in range(Nsources):
                    #     input_q = node_input_output.get(node.input[q], None)
                    #     k_srcs[q] = searchForLayer(input_q, layerParams)
                    # layerParams_k['source'] = k_srcs

                # Check if activation is defined within the conv layer  
               
               
                # next_node = output_to_next_node.get(node.output[0], None)
                # # print("next node: ", next_node.name)
                
                # if next_node and next_node.op_type in ['Relu', 'Tanh', 'Clip']:
                #     if next_node.op_type == 'Relu':
                #         activation = {}
                #         activation['name'] = f"{node.name}_{next_node.op_type.lower()}"
                #         activation['type'] = "RECTLINEAR"
                #         activation['bound'] = 1e20
                #         layerParams_k['activation'] = activation
                #     elif next_node.op_type == 'Clip':
                #         activation = {}
                #         activation['name'] = f"{node.name}_{next_node.op_type.lower()}"
                #         activation['type'] = "CLIP"
                #         activation['min'] = 0
                #         activation['max'] = 6
                #         layerParams_k['activation'] = activation
                #     elif next_node.op_type == 'Tanh':
                #         activation = {}
                #         activation['name'] = f"{node.name}_{next_node.op_type.lower()}"
                #         activation['type'] = "TANH"
                #         layerParams_k['activation'] = activation
                #     else:
                #         raise ValueError("Unrecognized activation in conv layer")
                # # else:
                #     print(f"Node: {node.name}, Activation: None")
            ###################
            ##    POOLING    ##
            ###################
            elif class_name in ('MaxPool', 'AveragePool', 'GlobalAveragePool', 'GlobalMaxPool'):

                layerParams_k['type'] = 'pool'
                if class_name in ('MaxPool', 'GlobalMaxPool'):
                    layerParams_k['poolType'] = 'max'
                elif class_name in ('AveragePool', 'GlobalAveragePool'):
                    layerParams_k['poolType'] = 'avg'

                # Pooling kernel size and stride: For global pooling, these will be set later
                if class_name not in ('GlobalAveragePool', 'GlobalMaxPool'):
                    for attr in node.attribute:
                        if attr.name == 'kernel_shape':
                            layerParams_k['MPx'], layerParams_k['MPy'] = attr.ints
                        elif attr.name == 'strides':
                            layerParams_k['stride_MP'] = attr.ints[0] # Assuming equal stride in all dimensions
                else:
                    layerParams_k['MPx'] = layerParams_k['MPy'] = 0 # Placeholder values
                    layerParams_k['stride_MP'] = 1

                # Padding information
                layerParams_k['py_L'] = layerParams_k['px_L'] = layerParams_k['py_R'] = layerParams_k['px_R'] = 0
                layerParams_k['padding'] = None

                for attr in node.attribute:
                    if attr.name == 'pads':
                        layerParams_k['py_L'], layerParams_k['px_L'], layerParams_k['py_R'], layerParams_k['px_R'] = attr.ints
                        # layerParams_k['padding'] = 'same'
                # Rounding option for average pooling
                # This needs a mechanism to check the next node's class name in ONNX format
                # For now, assuming it's not set, as there's no direct equivalent in ONNX
                layerParams_k['round'] = False

                # Find source layer: In ONNX, this is usually the input to the node
                if len(layerParams) > 0:
                    input1 = node_input_output.get(node.input[0], None)
                    if input1 is not None:
                        k_src = searchForLayer(input1, layerParams)
                    else:
                        k_src = len(layerParams) - 1
                    layerParams_k['source'] = np.array([k_src])
                    # Split before BN determination (if applicable)
                    layerParams_k['splitBeforeBN'] = ((k_src in layerParams) and (((layerParams[k_src]['type'] == 'add') and ('add' in input1)) or ((layerParams[k_src]['type'] == 'mul') and ('mul' in input1))))
                else:
                    layerParams_k['source'] = None  # First layer
            ###################
            ##     DENSE     ##
            ###################
            elif class_name in ('Gemm', 'MatMul','QLinearMatMul'):
                layerParams_k['type'] = 'dense'
                # Extracting the number of units and bias usage from ONNX attributes
                if quantized == False :
                    # for attr in node.attribute:
                        # if attr.name == 'transB':
                    weight_name = node.input[1]
                    layerParams_k['units'] = weights[weight_name].shape[0]  # Assuming weight shape [units, input_features]
                    layerParams_k['bias'] = len(node.input) == 3  # Bias exists if there are 3 inputs to the node
                    layerParams_k['binarizeWeights'] = False  # Placeholder, specific to Larq
                    
                    # if class_name == 'QLinearMatMul':
                    #     filter = [name for name in node.input if 'weight_quantized' in name]
                    # else :
                    #     filter = [name for name in node.input if 'weight' in name]
                    # layerParams_k['units'] = weights[filter[0]].shape[0]  # Assuming weight shape [units, input_features]
                    # layerParams_k['bias'] = any('bias' in name for name in node.input)  # Bias exists if there are 3 inputs to the node
                    # layerParams_k['binarizeWeights'] = False  # Placeholder, specific to Larq

                    # Find source layer
                    # If the layer is preceded by a flatten layer, find the source of that layer
                    if len(layerParams) > 0:
                        input1 = node_input_output.get(node.input[0], None)
                        if input1 is not None:
                            k_src = searchForLayer(input1, layerParams)
                        else:
                            k_src = len(layerParams) - 1
                        layerParams_k['source'] = np.array([k_src])
                        # Split before BN determination (if applicable)
                        layerParams_k['splitBeforeBN'] = ((k_src in layerParams) and (((layerParams[k_src]['type'] == 'add') and ('add' in input1)) or ((layerParams[k_src]['type'] == 'mul') and ('mul' in input1))))
                    else:
                        layerParams_k['source'] = None  # First layer
                else:
                    input_q = node_input_output.get(node.input[0], None)
                    input_0 = searchForLayer(input_q, layerParams)
                    layerParams_k['x_scale'] = layerParams[input_0]['scale'] 
                    layerParams_k['x_zero_point'] = layerParams[input_0]['zero_point']
                    layerParams[input_0]['needless'] = True

                    input_q = node_input_output.get(node.input[1], None)
                    input_1 = searchForLayer(input_q, layerParams)
                    layerParams_k['w_scale'] = layerParams[input_1]['scale'] 
                    layerParams_k['w_zero_point'] = layerParams[input_1]['zero_point']
                    layerParams[input_1]['needless'] = True


                    layerParams_k['bias'] = len(node.input) == 3  # Bias exists if there are 3 inputs to the node
                    layerParams_k['binarizeWeights'] = False  # Placeholder, specific to Larq

                    input_q = node_input_output.get(node.input[1], None)
                    input_1 = searchForLayer(input_q, layerParams)
                    layerParams_k['units'] = layerParams[input_1]['units']
                    
                    Nsources = len(node.input)
                    k_srcs = np.zeros(Nsources, dtype=int)
                    for q in range(Nsources):
                        input_q = node_input_output.get(node.input[q], None)
                        k_srcs[q] = searchForLayer(input_q, layerParams)
                    layerParams_k['source'] = k_srcs

                    
            #####################
            ##     FLATTEN     ##
            #####################
            elif class_name == 'Flatten':
                # Find source layers
                input1 = node_input_output.get(node.input[0], None)
                if input1 is not None:
                    k_src = searchForLayer(input1, layerParams)
                    layerParams_k['source'] = np.array([k_src])
                else:
                    k_src = len(layerParams) - 1
                    layerParams_k['source'] = np.array([k_src])
                layerParams_k['type'] = 'flatten'

            #####################
            ##     ReduceMean     ##
            #####################
            elif class_name == 'ReduceMean':
                # Find source layers
                input1 = node_input_output.get(node.input[0], None)
                if input1 is not None:
                    k_src = searchForLayer(input1, layerParams)
                    layerParams_k['source'] = np.array([k_src])
                else:
                    k_src = len(layerParams) - 1
                    layerParams_k['source'] = np.array([k_src])
                
                for attr in node.attribute:
                    if attr.name == 'axes':
                        layerParams_k['axes'] = attr.ints
                    elif attr.name == 'keepdims':
                        layerParams_k['keepdims'] = attr.i
                    

                layerParams_k['type'] = 'reducemean'

            #########################
            ##  ADD & CONCATENATE  ##
            #########################
            elif class_name in ('Add', 'Concat', 'Mul'):
                # Find source layers
                Nsources = len(node.input)
                k_srcs = np.zeros(Nsources, dtype=int)
                for q in range(Nsources):
                    input_q = node_input_output.get(node.input[q], None)
                    k_srcs[q] = searchForLayer(input_q, layerParams)
                layerParams_k['source'] = k_srcs

                if class_name == 'Add' :
                    layerParams_k['type'] = 'add'
                    # Split before BN determination (if applicable)
                    # This part might need to be adapted further based on the specific model structure
                    if Nsources == 2:
                        input1 = node_input_output.get(node.input[0], None)
                        input2 = node_input_output.get(node.input[1], None)
                        if layerParams[k_srcs[0]]['type'] == 'add':
                            layerParams_k['splitBeforeBN'] = ('add' in input1)
                        elif layerParams[k_srcs[1]]['type'] == 'add':
                            layerParams_k['splitBeforeBN'] = ('add' in input2)
                elif class_name == 'Mul' :
                    layerParams_k['type'] = 'mul'
                    # Split before BN determination (if applicable)
                    # This part might need to be adapted further based on the specific model structure
                    if Nsources == 2:
                        input1 = node_input_output.get(node.input[0], None)
                        input2 = node_input_output.get(node.input[1], None)
                        if layerParams[k_srcs[0]]['type'] == 'mul':
                            layerParams_k['splitBeforeBN'] = ('mul' in input1)
                        elif layerParams[k_srcs[1]]['type'] == 'mul':
                            layerParams_k['splitBeforeBN'] = ('mul' in input2)
                elif class_name == 'Concat':
                    layerParams_k['type'] = 'concat'
                    # In ONNX, Concat node has an 'axis' attribute
                    for attr in node.attribute:
                        if attr.name == 'axis' and attr.i not in [-1, 1]:
                            raise ValueError("Concatenation only supported along channel dimension")
            elif class_name in ('QLinearAdd'):
                # Find source layers
                k_srcs = [0,0]
                input_1 = node_input_output.get(node.input[0], None)
                if input_1 == None: 
                    layerParams_k['constant'] = True
                    k_srcs[0] = node.input[0]
                else:
                    k_srcs[0] = searchForLayer(input_1, layerParams)
                
                input_2 = node_input_output.get(node.input[3], None)
                if input_2 == None: 
                    layerParams_k['constant'] = True
                    k_srcs[1] = node.input[3]
                else:
                    k_srcs[1] = searchForLayer(input_2, layerParams)
                
                layerParams_k['source'] = k_srcs
                layerParams_k['type'] = 'add'
                layerParams_k['bias'] = any('bias' in name for name in node.input)
                if k_srcs[0] in layerParams and layerParams[k_srcs[0]]['type'] == 'add':
                    layerParams_k['splitBeforeBN'] = ('add' in input1)
                elif k_srcs[1] in layerParams and layerParams[k_srcs[1]]['type'] == 'add':
                    layerParams_k['splitBeforeBN'] = ('add' in input2)
                
            #################################
            ##  INT8 CUSTOM LAYERS         ##
            #################################
            elif class_name in ('QuantizeLinear','DequantizeLinear'):
                # Custom quantization/de-quantization layer used in certain networks
                if class_name == 'QuantizeLinear':  
                    layerParams_k['type'] = 'quantize'
                elif class_name == 'DequantizeLinear':
                    layerParams_k['type'] = 'dequantize'
                k_src = -1
                layerParams_k['has_input']=[]
                if 'Constant' in node.input[0]:
                    x_name = node.input[0]
                   
                    if len(weights[x_name].shape) == 4:
                        layerParams_k['Nox'] = weights[x_name].shape[2]
                        layerParams_k['Noy'] = weights[x_name].shape[3]
                        layerParams_k['Nic'] = weights[x_name].shape[1]
                        layerParams_k['Noc'] = weights[x_name].shape[0]
                        weight_for_quantized[node.output[0]] = (weights[node.input[0]] - weights[node.input[2]][:, np.newaxis, np.newaxis, np.newaxis] ) #* weights[node.input[1]].item()
                        # weight_for_quantized[node.output[0]] = weights[node.input[0]]
                        layerParams_k['zero_point'] = weights[node.input[2]]
                        layerParams_k['scale'] = weights[node.input[1]]
                        layerParams_k['needless'] = True
                    elif len(weights[x_name].shape) == 1:
                        layerParams_k['Noc'] = weights[x_name].shape[0]
                        weight_for_quantized[node.output[0]] = (weights[node.input[0]]) #* weights[node.input[1]].item()
                        # weight_for_quantized[node.output[0]] = weights[node.input[0]]
                        layerParams_k['zero_point'] = 0
                        layerParams_k['scale'] = weights[node.input[1]]
                        layerParams_k['needless'] = True
                    elif len(weights[x_name].shape) == 2:
                        layerParams_k['units'] = weights[x_name].shape[0]
                        weight_for_quantized[node.output[0]] = (weights[node.input[0]] - weights[node.input[2]][:, np.newaxis]) #* weights[node.input[1]].item()
                        # weight_for_quantized[node.output[0]] = weights[node.input[0]]
                        layerParams_k['zero_point'] = weights[node.input[2]]
                        layerParams_k['scale'] = weights[node.input[1]]
                        layerParams_k['needless'] = True
                    
                    append = False
                    
                else :
                    input1 = node_input_output.get(node.input[0], None)
                    layerParams_k['has_input'].append('x')
                    if input1 is not None:
                        k_src = searchForLayer(input1, layerParams)
                    else:
                        k_src = len(layerParams) - 1
                    
                    if 'Constant' in node.input[1]:
                        scale_name = node.input[1]
                        layerParams_k['scale'] = weights[scale_name].item()
                    else :
                        input1 = node_input_output.get(node.input[1], None)
                        layerParams_k['has_input'].append('scale')
                        if input1 is not None:
                            k_src = searchForLayer(input1, layerParams)
                        else:
                            k_src = len(layerParams) - 1

                    if 'Constant' in node.input[2]:
                        scale_name = node.input[2]
                        layerParams_k['zero_point'] = weights[scale_name].item()
                    else :
                        input1 = node_input_output.get(node.input[2], None)
                        layerParams_k['has_input'].append('zero_point')
                        if input1 is not None:
                            k_src = searchForLayer(input1, layerParams)
                        else:
                            k_src = len(layerParams) - 1

                    if k_src >= 0 and layerParams[k_src]['type'] in ("conv","dense"):
                        if layerParams[k_src]['activation'] is not None:
                            layerParams[k_src]['y_scale'] = layerParams_k['scale']
                            layerParams[k_src]['y_zero_point'] = layerParams_k['zero_point']
                            layerParams[k_src-1]['y_scale'] = layerParams_k['scale']
                            layerParams[k_src-1]['y_zero_point'] = layerParams_k['zero_point']
                        else :
                            layerParams[k_src]['y_scale'] = layerParams_k['scale']
                            layerParams[k_src]['y_zero_point'] = layerParams_k['zero_point']
                        
                        layerParams_k['needless'] = True


                layerParams_k['source'] = np.array([k_src])
            
            #################################
            ##  Cast LAYERS         ##
            #################################
            elif class_name in ('Cast'):
                # Custom quantization/de-quantization layer used in certain networks
                layerParams_k['type'] = 'cast'

                for attr in node.attribute:
                    if attr.name == 'to':
                        layerParams_k['casttype'] = onnx.TensorProto.DataType.Name(attr.i)
                
                # Find source layer
                input1 = node_input_output.get(node.input[0], None)
                if input1 is not None:
                    k_src = searchForLayer(input1, layerParams)
                else:
                    k_src = len(layerParams) - 1
                layerParams_k['source'] = np.array([k_src])
                
                if 'ConstantOfShape' in node.input[0]:
                    layerParams_k['needless'] = True

            #################################
            ##  ConstantOfShape LAYERS         ##
            #################################
            elif class_name in ('ConstantOfShape'):
                # Custom quantization/de-quantization layer used in certain networks
                layerParams_k['type'] = 'constant_shape'

                for attr in node.attribute:
                    if attr.name == 'value':
                        layerParams_k['value'] = onnx.numpy_helper.to_array(attr.t)
                
                shape_name = node.input[0]
                layerParams_k['shape'] = weights[shape_name].item()
                # Find source layer
                if 'Constant' in node.input[0]:
                    k_src = -1
                else:
                    input1 = node_input_output.get(node.input[0], None)
                    if input1 is not None:
                        k_src = searchForLayer(input1, layerParams)
                    else:
                        k_src = len(layerParams) - 1
                layerParams_k['needless'] = True
                layerParams_k['source'] = np.array([k_src])
            
            layerParams.append(layerParams_k)


        elif isActivation(class_name) or class_name in ignoredLayerTypes:
            node_input_output[node.output[0]] = node.name
            # Find source layer
            input1 = node_input_output.get(node.input[0], None)
            if input1 is not None:
                k_src = searchForLayer(input1, layerParams)
            else:
                k_src = len(layerParams) - 1
            with_activate = layerParams[k_src].copy()
            
            # print(layerParams)
            # Reshape is only implemented in the trivial case where it is equivalent to np.squeeze()
            if class_name == "Reshape":                
                shape_attribute = next((attr for attr in node.attribute if attr.name == 'shape'), None)
                if shape_attribute:
                    target_shape = np.array([dim for dim in shape_attribute.ints])
                    if np.sum(target_shape > 1) > 1:
                        raise ValueError("Reshape layer is unimplemented, except in the trivial case")


            # # Batchnorm: associate parameters with previous layer
            # elif node.op_type == 'BatchNormalization':
            #     # BatchNormalization 노드의 속성을 추출
            #     bn_params = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            #     # epsilon, scale, center 값 추출
            #     epsilon = bn_params.get('epsilon', None)
            #     scale = bn_params.get('scale', None)
            #     center = bn_params.get('center', None)  
            #     layerParams[k_src]['batch_norm'] = node_name
            #     layerParams[k_src]['epsilon'] = bn_params.get('epsilon', None)
            #     layerParams[k_src]['BN_scale'] = bn_params.get('scale', None)
            #     layerParams[k_src]['BN_center'] = bn_params.get('center', None)            

            if class_name in ignoredLayerTypes:
                with_activate['appended'] = node_name

            # Bind the activation to the relevant layer
            elif isActivation(class_name):
                activation = {}
                activation['name'] = node_name
                activation['output_name'] = node.output[0]
                if class_name == "Relu": 
                    activation['type'] = "RECTLINEAR"
                    activation['bound'] = 1e20
                elif class_name == 'Clip':
                    activation['type'] = "CLIP"
                    # for input in node.input:
                        # print("Input name:", input)
                    activation['min'] = 0
                    activation['max'] = 6
                elif class_name == "Sigmoid": 
                    activation['type'] = "SIGMOID"
                elif class_name == "Softmax": 
                    activation['type'] = "SOFTMAX"
                with_activate['activation'] = activation
            
            #if you want to use skipconnention without activation, you can use this code
            if layerParams[k_src]["type"]== 'conv' and False:
                layerParams.append(with_activate)
            else:
               layerParams[k_src] = with_activate
        else:
            if class_name != "Constant":
                raise ValueError("Unrecognized Keras layer type "+class_name)
    layerParams.append(layerParams_first)
     

    for layer in layerParams:
        if layer['type'] == 'add':
            print(layer)
        # 디버그 정보 출력
    output_node_name=[]
    for layer in layerParams:
        if search_accuracy :
            if quantized:
                if layer['type'] == 'quantize':
                    if k_src >= 0 and layerParams[layer["source"][0]]['type'] in ("conv","dense"):
                        # if layer['output_name'] != 'output':
                        output_node_name.append(layer['output_name'])

                            
            else:
                if layer['type'] == 'conv' or layer['type'] == 'dense':
                    if layer['activation'] is not None:
                        if layer['activation']['output_name'] != 'output':
                            output_node_name.append(layer['activation']['output_name'])
                    else:
                        if layer['output_name'] != 'output':
                            output_node_name.append(layer['output_name'])
    navcim_dir = os.getenv('NAVCIM_DIR')
    if search_accuracy and os.path.exists(f'{navcim_dir}/cross-sim/applications/dnn/inference/{model_name}_add_output.onnx') == False:      
        value_info_protos = []
        shape_info = onnx.shape_inference.infer_shapes(model)
        for idx, node in enumerate(shape_info.graph.value_info):
            if node.name in output_node_name:
                value_info_protos.append(node)
        assert len(value_info_protos) == len(output_node_name)
        model.graph.output.extend(value_info_protos)  #  in inference stage, these tensor will be added to output dict.
        onnx.checker.check_model(model)
        onnx.save(model, f'{navcim_dir}/cross-sim/applications/dnn/inference/{model_name}_add_output.onnx')
    output_node_name.append('output')
        # if debug_graph:
            # print(f'Node Name: {node_name}, Type: {class_name}')
            # print(f'Input Tensors: {input_tensors}')
            # print(f'Output Tensors: {output_tensors}')
            # 추가 디버그 정보
            # ...
    
    sizes = [None for i in range(len(layerParams)+1)]
    for j in range(len(layerParams)-1):
        if j >= 0:
            j_src = layerParams[j]['source'][0]
            if layerParams[j]['type'] == 'conv':
                # Caclulate input fmap size for a layer that is not the first in the network
                if j != 0:
                    # if quantized == True :
                    #     layerParams[j]['Noc'] = layerParams[layerParams[j]['source'][1]]['Noc'] 
                    layerParams[j]['Nix'] = layerParams[j_src]['Nox']
                    layerParams[j]['Niy'] = layerParams[j_src]['Noy']
                    layerParams[j]['Nic'] = layerParams[j_src]['Noc']
                    
                    if layerParams[j]['depthwise']:
                        layerParams[j]['Noc'] = layerParams[j]['Nic']
                
                    


                sizes[j] = (layerParams[j]['Nix'], layerParams[j]['Niy'], layerParams[j]['Nic'])

                # Compute output feature map size
                if layerParams[j]['sameConv']:
                    layerParams[j]['Nox'] = layerParams[j]['Nix']//layerParams[j]['stride']
                    layerParams[j]['Noy'] = layerParams[j]['Niy']//layerParams[j]['stride']
                else:
                    layerParams[j]['Nox'] = 1 + (layerParams[j]['Nix']-layerParams[j]['Kx']+layerParams[j]['px_0']+layerParams[j]['px_1'])//layerParams[j]['stride']
                    layerParams[j]['Noy'] = 1 + (layerParams[j]['Niy']-layerParams[j]['Ky']+layerParams[j]['py_0']+layerParams[j]['py_1'])//layerParams[j]['stride']

                # Occasionally, conv is the final layer
                if j == len(layerParams)-2:
                    sizes[j+1] = (layerParams[j]['Nox'], layerParams[j]['Noy'], layerParams[j]['Noc'])

            elif layerParams[j]['type'] == 'add':
                if 'units' in layerParams[j_src]:
                    sizes[j] = sizes[j_src]
                    layerParams[j]['units'] = layerParams[j_src]['units']
                else:
                    Nsources = len(layerParams[j]['source'])
                    size0 = (layerParams[j_src]['Nox'], layerParams[j_src]['Noy'], layerParams[j_src]['Noc'])
                    for q in range(1,Nsources):
                        j_src_q = layerParams[j]['source'][q]
                        size_q = (layerParams[j_src_q]['Nox'], layerParams[j_src_q]['Noy'], layerParams[j_src_q]['Noc'])
                        if size0 != size_q: raise ValueError("Incoming feature map dimensions to Add layer do not match")
                    sizes[j] = size0
                    layerParams[j]['Nox'] = layerParams[j_src]['Nox']
                    layerParams[j]['Noy'] = layerParams[j_src]['Noy']
                    layerParams[j]['Noc'] = layerParams[j_src]['Noc']
                    
            elif layerParams[j]['type'] == 'mul':
                if 'units' in layerParams[j_src]:
                    sizes[j] = sizes[j_src]
                    layerParams[j]['units'] = layerParams[j_src]['units']
                else:
                    Nsources = len(layerParams[j]['source'])
                    size0 = (layerParams[j_src]['Nox'], layerParams[j_src]['Noy'], layerParams[j_src]['Noc'])
                    j_src_1 = layerParams[j]['source'][1]
                    size1 = (layerParams[j_src_1]['Nox'], layerParams[j_src_1]['Noy'], layerParams[j_src_1]['Noc'])

                    
                    if size0 != size1: 
                        layerParams[j]['se_block'] = True
                        if size0[0] > size1[0]:
                            sizes[j] = size0
                            layerParams[j]['Nox'] = layerParams[j_src]['Nox']
                            layerParams[j]['Noy'] = layerParams[j_src]['Noy']
                            layerParams[j]['Noc'] = layerParams[j_src]['Noc']
                        else:
                            sizes[j] = size1
                            layerParams[j]['Nox'] = layerParams[j_src_1]['Nox']
                            layerParams[j]['Noy'] = layerParams[j_src_1]['Noy']
                            layerParams[j]['Noc'] = layerParams[j_src_1]['Noc']
                    else:
                        layerParams[j]['se_block'] = False
                        sizes[j] = size0
                        layerParams[j]['Nox'] = layerParams[j_src]['Nox']
                        layerParams[j]['Noy'] = layerParams[j_src]['Noy']
                        layerParams[j]['Noc'] = layerParams[j_src]['Noc']

            elif layerParams[j]['type'] == 'concat':
                Nsources = len(layerParams[j]['source'])

                if layerParams[j_src]['type'] in ("conv","pool","add","concat","quantize"):
                    size0 = (layerParams[j_src]['Nox'], layerParams[j_src]['Noy'], layerParams[j_src]['Noc'])
                elif layerParams[j_src]['type'] in ("dense","flatten","flatten_input","scale"):
                    size0 = (1,1, layerParams[j_src]['units'])
        
                Noc_out = size0[2]
                for q in range(1,Nsources):
                    j_src_q = layerParams[j]['source'][q]
                    if layerParams[j_src_q]['type'] in ("conv","pool","add","concat","quantize"):
                        size_q = (layerParams[j_src_q]['Nox'], layerParams[j_src_q]['Noy'], layerParams[j_src_q]['Noc'])
                        if (size0[0] != size_q[0]) or (size0[1] != size_q[1]):
                            raise ValueError("Incoming feature map dimensions to Concat layer do not match")
                    elif layerParams[j_src_q]['type'] in ("dense","flatten","flatten_input","scale"):
                        size_q = (1,1,layerParams[j_src_q]['units'])

                    Noc_out += size_q[2]
            
                if layerParams[j_src]['type'] in ("conv","pool","add","concat","quantize"):
                    sizes[j] = (layerParams[j_src]['Nox'], layerParams[j_src]['Noy'], Noc_out)
                    layerParams[j]['Nox'] = layerParams[j_src]['Nox']
                    layerParams[j]['Noy'] = layerParams[j_src]['Noy']
                    layerParams[j]['Noc'] = Noc_out
                elif layerParams[j_src]['type'] in ("dense","flatten","flatten_input","scale"):
                    sizes[j] = (1,1, Noc_out)
                    layerParams[j]['units'] = Noc_out

            elif layerParams[j]['type'] == 'pool':
                sizes[j] = (layerParams[j_src]['Nox'], layerParams[j_src]['Noy'], layerParams[j_src]['Noc'])
                layerParams[j]['Noc'] =  layerParams[j_src]['Noc']

                # Detect if pooling type is global
                if layerParams[j]['MPx'] == 0 or layerParams[j]['MPy'] == 0:
                    layerParams[j]['MPx'] = sizes[j][0]
                    layerParams[j]['MPy'] = sizes[j][1]
                
                # Handle same padding in pooling layers
                # The code below is functionally identical to the padding logic in convolution_parameters.py
                MPx = layerParams[j]['MPx']
                MPy = layerParams[j]['MPy']
                Nix = layerParams[j_src]['Nox']
                Niy = layerParams[j_src]['Noy']
                stride = layerParams[j]['stride_MP']
                if layerParams[j]['padding'] == 'same':
                    layerParams[j]['Nox'] = Nix // stride
                    layerParams[j]['Noy'] = Niy // stride
                    if (MPx % 2 != 0) and (MPy % 2 != 0):
                        # Odd size filter
                        if (Nix % stride == 0):
                            px = max(MPx - stride, 0)
                        else:
                            px = max(MPx - (Nix % stride), 0)
                        if (Niy % stride == 0):
                            py = max(MPy - stride, 0)
                        else:
                            py = max(Ky - (Niy % stride), 0)
                    else:
                        # Even size filter
                        px = (layerParams[j]['Nox'] - 1)*stride + MPx - Nix
                        py = (layerParams[j]['Noy'] - 1)*stride + MPy - Niy
                    layerParams[j]['px_L'] = px // 2
                    layerParams[j]['px_R'] = px - layerParams[j]['px_L']
                    layerParams[j]['py_L'] = py // 2
                    layerParams[j]['py_R'] = py - layerParams[j]['py_L']

                else:
                    # This is used for valid padding and ZeroPadding2D
                    layerParams[j]['Nox'] = 1 + (Nix-MPx+layerParams[j]['px_L']+layerParams[j]['px_R']) // stride
                    layerParams[j]['Noy'] = 1 + (Niy-MPy+layerParams[j]['py_L']+layerParams[j]['py_R']) // stride

                if j == len(layerParams)-2:
                    sizes[j+1] = (layerParams[j]['Nox'], layerParams[j]['Noy'], layerParams[j]['Noc'])
                

            elif layerParams[j]['type'] == 'flatten' :
                if 'units' in layerParams[j_src]:
                    sizes[j] = (1,1,layerParams[j_src]['units'])
                elif layerParams[j_src]['type'] in ("conv","pool","add","quantize","dequantize"):
                    sizes[j] = (1,1,layerParams[j_src]['Nox']*layerParams[j_src]['Noy']*layerParams[j_src]['Noc'])
                layerParams[j]['units'] = sizes[j][0]*sizes[j][1]*sizes[j][2]

            elif layerParams[j]['type'] == 'reducemean' :
                if 'units' in layerParams[j_src]:
                    sizes[j] = (1,1,layerParams[j_src]['units'])
                elif layerParams[j_src]['type'] in ("conv","pool","add","quantize","dequantize"):
                    tmp = layerParams[j_src]['Nox']*layerParams[j_src]['Noy']*layerParams[j_src]['Noc']
                    if 1 in layerParams[j]['axes']:
                        tmp = tmp / layerParams[j_src]['Noc']
                    if 2 in layerParams[j]['axes']:
                        tmp = tmp / layerParams[j_src]['Noy']
                    if 3 in layerParams[j]['axes']:
                        tmp = tmp / layerParams[j_src]['Nox']
                    sizes[j] = (1,1,int(tmp))
                layerParams[j]['units'] = sizes[j][0]*sizes[j][1]*sizes[j][2]

            elif layerParams[j]['type'] == 'flatten_input':
                sizes[j] = (Nix0,Niy0,Nic0)
                layerParams[j]['units'] = sizes[j][0]*sizes[j][1]*sizes[j][2]

            elif layerParams[j]['type'] == 'dense':
                if j != 0:
                    if layerParams[j_src]['type'] in ("conv","pool","add"):
                        sizes[j] = (1,1,layerParams[j_src]['Nox']*layerParams[j_src]['Noy']*layerParams[j_src]['Noc'])
                        # layerParams[j]['units'] = layerParams[j_src]['Nox']*layerParams[j_src]['Noy']*layerParams[j_src]['Noc']
                    else:
                        sizes[j] = (1,1,layerParams[j_src]['units'])

                else:
                    sizes[j] = (1,1,Nix0) # Dense is first layer: input must be 1D

                if j == len(layerParams)-2:
                    sizes[j+1] = (1,1,layerParams[j]['units'])

            elif layerParams[j]['type'] in ("quantize","dequantize"):
                # This assumes no quantization layers are present after flattening
                if 'x' not in layerParams[j]['has_input']:
                    if 'units' in layerParams[j]:
                        sizes[j] = (1,1,layerParams[j]['units'])
                    elif 'Nox' in layerParams[j]:
                        sizes[j] = (layerParams[j]['Nox'], layerParams[j]['Noy'], layerParams[j]['Noc'])
                    else :
                        sizes[j] = (1, 1, layerParams[j]['Noc'])
                else:
                    if 'units' in layerParams[j_src]:
                        layerParams[j]['units'] = layerParams[j_src]['units']
                        sizes[j] = (1,1,layerParams[j]['units'])
                    else:    
                        sizes[j] = (layerParams[j_src]['Nox'], layerParams[j_src]['Noy'], layerParams[j_src]['Noc'])
                        layerParams[j]['Nox'] = layerParams[j_src]['Nox']
                        layerParams[j]['Noy'] = layerParams[j_src]['Noy']
                        layerParams[j]['Noc'] = layerParams[j_src]['Noc']
            elif layerParams[j]['type'] in ("cast"):
                # This assumes no quantization layers are present after flattening
                if 'units' in layerParams[j_src]:
                    layerParams[j]['units'] = layerParams[j_src]['units']
                else:    
                    sizes[j] = (layerParams[j_src]['Nox'], layerParams[j_src]['Noy'], layerParams[j_src]['Noc'])
                    layerParams[j]['Nox'] = layerParams[j_src]['Nox']
                    layerParams[j]['Noy'] = layerParams[j_src]['Noy']
                    layerParams[j]['Noc'] = layerParams[j_src]['Noc']

            elif layerParams[j]['type'] == 'constant_shape':
                # This assumes no scale layers are present before flattening
                sizes[j] = (1,1,1)
                layerParams[j]['Nox'] = 1
                layerParams[j]['Noy'] = 1
                layerParams[j]['Noc'] = 1

            elif layerParams[j]['type'] == 'scale':
                # This assumes no scale layers are present before flattening
                sizes[j] = (1,1,layerParams[j_src]['units'])
                if j == len(layerParams)-2:
                    sizes[j+1] = (1,1,layerParams[j]['units'])

        # For debug
            
        if debug_graph and False:
            print('Active layer: '+layerParams[j]['name'])
            if layerParams[j]['type'] == 'conv':
                print('     Kernel: '+str(layerParams[j]['Kx'])+' x '+str(layerParams[j]['Ky']))
                print('     Channels: '+str(layerParams[j]['Noc'])+' x '+str(layerParams[j]['Noc']))
                print('     Strides:'+str(layerParams[j]['stride'])+' x '+str(layerParams[j]['stride']))
            if layerParams[j]['source'] is not None:
                print('   Source layer 1: '+layerParams[layerParams[j]['source'][0]]['name'])
                if len(layerParams[j]['source']) > 1:
                    if layerParams[j]['constant']:
                        print('   Source layer 2: '+layerParams[j]['source'][1])
                    else:
                        print('   Source layer 2: '+layerParams[layerParams[j]['source'][1]]['name'])
                print('   Take pure add as input: '+str(layerParams[j]['splitBeforeBN']))
            if layerParams[j]['batch_norm'] is not None:
                print('   Batchnorm layer: '+layerParams[j]['batch_norm'])
            else:
                print('   Batchnorm layer: None')
            if layerParams[j]['activation'] is not None:
                print('   Activation layer: '+layerParams[j]['activation']['type'])
            else:
                print('   Activation layer: None')
        # print(j,layerParams[j])
    # for i in layerParams:
        # print(i)      
        
    # 모델의 텐서 정보 분석
    for tensor in tensors:
        tensor_name = tensor.name
        tensor_data = numpy_helper.to_array(tensor)
        # 텐서 관련 추가 정보 처리
        # ...

    # 필요한 추가 분석
    # ...

    # 결과 반환
    layerParams.pop(-1)
    sizes.pop(-1)
    filtered_layerParams, filtered_size = zip(*[(d, s) for d, s in zip(layerParams, sizes) if not d.get('needless', False)])

    # Convert the filtered results back to lists
    # filtered_layerParams = list(filtered_layerParams)
    # filtered_size = list(filtered_size)

    return layerParams, sizes ,weight_for_quantized , output_node_name
# model = onnx.load("/root/models/vision/classification/vgg/model/vgg16-12-int8.onnx")
# metadata, size= get_onnx_metadata(model, debug_graph=True)

# print(metadata)
# print(size)