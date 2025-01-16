import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import ImageFile, Image

def set_seed(seed_value=42):
    """设置所有随机种子以确保实验的可重复性。"""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class ParametersDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        angle,
        transform=None,
        image_dim=(320, 320),
        flow_rate=True,
        feed_rate=True,
    ):
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.angle = angle
        self.transform = transform
        self.image_dim = image_dim
        self.use_flow_rate = flow_rate
        self.use_feed_rate = feed_rate

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.img_path.iloc[idx]
        if not img_path.lower().endswith('.jpg'):
            img_path += '.jpg'

        img_name = os.path.join(self.root_dir, self.angle, img_path)

        if not os.path.exists(img_name):
            print(f"File not found: {img_name}")
            return None

        dim = self.image_dim[0] / 2

        if self.angle == '45':
            left = self.dataframe.nozzle_tip_x_45[idx] - dim
            top = self.dataframe.nozzle_tip_y_45[idx] - dim
            right = self.dataframe.nozzle_tip_x_45[idx] + dim
            bottom = self.dataframe.nozzle_tip_y_45[idx] + dim
        elif self.angle == '90':
            left = self.dataframe.nozzle_tip_x_90[idx] - dim
            top = self.dataframe.nozzle_tip_y_90[idx] - dim
            right = self.dataframe.nozzle_tip_x_90[idx] + dim
            bottom = self.dataframe.nozzle_tip_y_90[idx] + dim
        elif self.angle == 'besides':
            left = self.dataframe.nozzle_tip_x_besides[idx] - dim
            top = self.dataframe.nozzle_tip_y_besides[idx] - dim
            right = self.dataframe.nozzle_tip_x_besides[idx] + dim
            bottom = self.dataframe.nozzle_tip_y_besides[idx] + dim
        else:
            raise ValueError(f"Unsupported angle: {self.angle}")

        image = Image.open(img_name).convert('RGB')
        image = image.crop((left, top, right, bottom))

        if self.transform:
            image = self.transform(image)

        flow_rate_class = int(self.dataframe.flow_rate_class[idx])
        feed_rate_class = int(self.dataframe.feed_rate_class[idx])

        # label = torch.zeros(9)
        # label[flow_rate_class * 3 + feed_rate_class] = 1

        # 新的标签映射
        old_label = flow_rate_class * 3 + feed_rate_class
        if old_label == 2:
            new_label = 0  # 严重挤出不足
        elif old_label in [0, 1, 5]:
            new_label = 1  # 轻微挤出不足
        elif old_label == 4:
            new_label = 2  # 正常
        elif old_label in [3, 7, 8]:
            new_label = 3  # 轻微过度挤出
        elif old_label == 6:
            new_label = 4  # 严重过度挤出
        else:
            raise ValueError(f"Invalid label combination: {old_label}")

        label = torch.zeros(5)
        label[new_label] = 1

        return image, label