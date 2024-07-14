import os
import numpy as np
from PIL import Image
from torchvision import transforms
from imagenet_path import imagenet_val_path  # ImageNet 검증 데이터셋의 경로

# 이미지 전처리 설정
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# 이미지 파일 이름 불러오기
fns0 = os.listdir(imagenet_val_path)
fns0.sort()
fns = [os.path.join(imagenet_val_path, fn) for fn in fns0]

# 데이터 저장을 위한 설정
output_dir = "./"  # 출력 디렉토리 설정
num_images = 500  # 처리할 이미지 수
x_val = np.zeros((num_images, 3, 224, 224), dtype=np.float32)

# 이미지 처리 및 저장
for i in range(num_images):
    img = Image.open(fns[i])
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = preprocess(img)
    img = img.numpy()

    # 이미지 저장
    x_val[i, :, :, :] = img

# .npy 파일로 저장
np.save(os.path.join(output_dir, "x.npy"), x_val)