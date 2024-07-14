import onnxruntime as ort
import numpy as np
import argparse
import os

navcim_dir = os.getenv('NAVCIM_DIR')

# Numpy 배열 로드
x_val = np.load(f'{navcim_dir}/cross-sim/applications/dnn/data/datasets/imagenet/x_val_torch_1000.npy')[:500]
y_val = np.load(f'{navcim_dir}/cross-sim/applications/dnn/data/datasets/imagenet/y_val.npy')[:500]
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--model', default='MobileNetV2', help='SqueezeNet|ResNet50')
args = parser.parse_args()

ort_session = ort.InferenceSession(f"{navcim_dir}/cross-sim/applications/dnn/inference/model/{args.model}.onnx")

# 모델 평가
correct = 0
total = 0
for i in range(x_val.shape[0]):
    # 입력 이미지 준비
    inputs = x_val[i:i+1]  # i번째 이미지를 하나의 배치로 만듦
    ort_inputs = {ort_session.get_inputs()[0].name: inputs}
    
    # ONNX 모델 추론
    ort_outs = ort_session.run(None, ort_inputs)
    predicted = np.argmax(ort_outs[0], axis=1)

    # 정확도 계산
    correct += (predicted == y_val[i]).sum()
    total += 1

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy}%')


