import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn
import argparse
import os

navcim_dir = os.getenv('NAVCIM_DIR')

from torchvision.models.quantization import MobileNet_V2_QuantizedWeights
# 모델과 데이터 준비
path = f"{navcim_dir}/cross-sim/applications/dnn/data/datasets/imagenet/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model = model.to(device)
model = nn.DataParallel(model)
model.eval()

x_test = np.load(path + "x_val_torch_1000.npy")
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
x_test_tensor = x_test_tensor[:2]
target_path = f"{navcim_dir}/cross-sim/applications/dnn/data/datasets/imagenet/y_val.npy"
y_target = np.load(target_path)

# x_test_tensor에 해당하는 앞에서부터 n개의 타겟 레이블 사용
n = x_test_tensor.size(0)  # 테스트 이미지의 개수

target = torch.tensor(y_target[:n], dtype=torch.long) 

# 손실 함수
x_test_tensor = x_test_tensor.to(device)
target = target.to(device)
loss_fn = torch.nn.CrossEntropyLoss()



params = list(model.parameters())
v = [torch.randn_like(p) for p in params]  # 임의의 방향 벡터 초기화
iterations = 100  # 반복 횟수

count = 0
hessian = []
for layer in model.modules():
    if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d) and layer.groups == 1:
        params = list(layer.parameters())
        if not params:  # 파라미터가 없는 레이어는 건너뜀
            continue
        
        # 각 레이어의 파라미터에 대한 Hessian 고유값 근사 로직
        v = [torch.randn_like(p).to(device) for p in params] 
        for i in range(iterations):
            model.zero_grad()
            output = model(x_test_tensor)
            loss = loss_fn(output, target)
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            Hv = torch.autograd.grad(grads, params, grad_outputs=v, only_inputs=True, retain_graph=True)
            
            v = [hv.detach() for hv in Hv]
            norm = torch.sqrt(sum(torch.dot(v_.reshape(-1), v_.reshape(-1)) for v_ in v))
            v = [v_ / norm for v_ in v]
        
        lambda_approx = sum(torch.dot(hv.reshape(-1), v_.reshape(-1)) for hv, v_ in zip(Hv, v)) / sum(torch.dot(v_.reshape(-1), v_.reshape(-1)) for v_ in v)
        print(f"Layer_{count}: {type(layer).__name__}, Approximate largest eigenvalue: {lambda_approx.item()}")
        hessian.append(lambda_approx.item())
        count += 1

#store list
np.save(f'{args.model}_hessian_list.txt', hessian)
