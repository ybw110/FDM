import os
from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import ImageFile, Image
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ParametersDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        angle,

        image_dim=(320, 320),
        pre_crop_transform=None,
        post_crop_transform=None,

        flow_rate=False,
        feed_rate=False,
        z_offset=False,
        hotend=False,
        actual_bed=False,  # 新增参数

        per_img_normalisation=False,
        dataset_name=None,
    ):
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.angle = angle
        self.dataset_name = dataset_name  # 添加这行

        self.pre_crop_transform = pre_crop_transform
        self.post_crop_transform = post_crop_transform

        self.image_dim = image_dim

        self.use_flow_rate = flow_rate
        self.use_feed_rate = feed_rate
        self.use_z_offset = z_offset
        self.use_hotend = hotend
        self.use_actual_bed = actual_bed  # 新增属性

        self.per_img_normalisation = per_img_normalisation

        self.targets = []

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        self.targets = []
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.img_path.iloc[idx]
        if not img_path.lower().endswith('.jpg'):
            img_path += '.jpg'

        img_name = os.path.join(self.root_dir, self.angle, img_path)

        if not os.path.exists(img_name):
            print(f"File not found: {img_name}")


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

        image = Image.open(img_name)
        if self.pre_crop_transform:
            image = self.pre_crop_transform(image)
        image = image.crop((left, top, right, bottom))

        if self.per_img_normalisation:
            tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
            image = tfms(image)
            mean = torch.mean(image, dim=[1, 2])
            std = torch.std(image, dim=[1, 2])
            image = transforms.Normalize(mean, std)(image)
        else:
            if self.post_crop_transform:
                image = self.post_crop_transform(image)

        if self.use_flow_rate:
            flow_rate_class = int(self.dataframe.flow_rate_class[idx])
            self.targets.append(flow_rate_class)

        if self.use_feed_rate:
            feed_rate_class = int(self.dataframe.feed_rate_class[idx])
            self.targets.append(feed_rate_class)

        if self.use_z_offset:
            z_offset_class = int(self.dataframe.z_offset_class[idx])
            self.targets.append(z_offset_class)

        if self.use_hotend:
            hotend_class = int(self.dataframe.hotend_class[idx])
            self.targets.append(hotend_class)

        if self.use_actual_bed:
            bed_valu_class = float(self.dataframe.actual_bed_class[idx])
            self.targets.append(bed_valu_class)

        y = torch.tensor(self.targets, dtype=torch.long)
        sample = (image, y)
        return sample
