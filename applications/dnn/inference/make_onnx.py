import torchvision.models as models
import torch
import argparse
import os

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--model', default='MobileNetV2', help='SqueezeNet|ResNet50')
args = parser.parse_args()

if args.model == 'MobileNetV2':
    model = models.quantization.mobilenet_v2(weights='DEFAULT',quantize=True)
elif args.model == 'ResNet50':
    model = models.quantization.resnet50(weights='DEFAULT',quantize=True)
elif args.model == 'SqueezeNet':
    model= models.squeezenet1_1(weights='DEFAULT')
else:
    raise ValueError("Invalid model name")

model.eval()

# for name, module in model.named_modules():
    # print(name, module)
# exit()
# there is no model folder make it
if not os.path.exists("./model"):
    os.mkdir("./model")

x = torch.randn(1,3,224,224)
torch_out = model(x)
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  f"./model/{args.model}.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=13,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  # training=TrainingMode.TRAINING,
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'])



