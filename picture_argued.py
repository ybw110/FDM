import os
import torch
import random
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)  # 或任何其他固定的数字

def to_pil_image(img):
    if isinstance(img, torch.Tensor):
        return transforms.ToPILImage()(img)
    return img

class StepByStepTransform:
    def __init__(self):
        self.transforms = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.4, contrast=0.8, saturation=0.8)
        ]
        self.step_images = []

    def __call__(self, img):
        self.step_images = [img]  # 原图
        for t in self.transforms:
            img = t(img)
            self.step_images.append(to_pil_image(img))
        return img

def main():
    print("hello")
    # 设置保存路径
    save_folder = 'augmentation_results'  # 指定保存文件夹的名称
    os.makedirs(save_folder, exist_ok=True)  # 创建文件夹（如果不存在）

    # 假设我们有一个图像文件
    image_path = '/data1/ybw/3D/ybw_data_3cam/my_picture_flow_feed/merge_data3/besides/image-0350.jpg'  # 请替换为实际路径
    original_image = Image.open(image_path)

    # 对于每个视角
    angles = ['besides']
    titles = ['Original', 'RandomResizedCrop', 'RandomHorizontalFlip',
              'RandomRotation', 'ColorJitter']

    for angle in angles:
        transform = StepByStepTransform()
        _ = transform(original_image)

        for i, img in enumerate(transform.step_images):
            save_path = os.path.join(save_folder, f'{angle}_{titles[i]}.png')
            img.save(save_path)
            print(f"已保存 {angle} 视角的 {titles[i]} 图像。")


if __name__ == '__main__':
    main()
