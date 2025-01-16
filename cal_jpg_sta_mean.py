import os
import pandas as pd
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, csv_file, root_dir, angle, image_dim=(320, 320), transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.angle_dir = angle
        self.transform = transform
        self.image_dim = image_dim
        self.angle = angle
        self.valid_indices = self._filter_valid_indices()

    def _filter_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.dataframe)):
            img_name = f"image-{self.dataframe.img_num.iloc[idx]:04d}.jpg"
            img_path = os.path.join(self.root_dir, self.angle_dir, img_name)
            if os.path.exists(img_path):
                valid_indices.append(idx)
            else:
                print(f"Warning: Image not found: {img_path}")
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        img_name = f"image-{self.dataframe.img_num.iloc[idx]:04d}.jpg"
        img_path = os.path.join(self.root_dir, self.angle_dir, img_name)

        image = Image.open(img_path)

        dim = self.image_dim[0] / 2
        if self.angle == '45':
            left = self.dataframe.nozzle_tip_x_45.iloc[idx] - dim
            top = self.dataframe.nozzle_tip_y_45.iloc[idx] - dim
            right = self.dataframe.nozzle_tip_x_45.iloc[idx] + dim
            bottom = self.dataframe.nozzle_tip_y_45.iloc[idx] + dim
        elif self.angle == '90':
            left = self.dataframe.nozzle_tip_x_90.iloc[idx] - dim
            top = self.dataframe.nozzle_tip_y_90.iloc[idx] - dim
            right = self.dataframe.nozzle_tip_x_90.iloc[idx] + dim
            bottom = self.dataframe.nozzle_tip_y_90.iloc[idx] + dim
        elif self.angle == 'besides':
            left = self.dataframe.nozzle_tip_x_besides.iloc[idx] - dim
            top = self.dataframe.nozzle_tip_y_besides.iloc[idx] - dim
            right = self.dataframe.nozzle_tip_x_besides.iloc[idx] + dim
            bottom = self.dataframe.nozzle_tip_y_besides.iloc[idx] + dim

        image = image.crop((left, top, right, bottom))

        if self.transform:
            image = self.transform(image)

        return image


def calculate_mean_std(dataloader, device):
    mean_sum = 0
    std_sum = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, total=len(dataloader), desc="Processing")

    for batch in progress_bar:
        batch = batch.to(device)
        batch_mean = torch.mean(batch, dim=[0, 2, 3])
        batch_std = torch.std(batch, dim=[0, 2, 3])

        batch_size = batch.size(0)
        total_samples += batch_size

        mean_sum += batch_mean * batch_size
        std_sum += batch_std * batch_size

    overall_mean = mean_sum / total_samples
    overall_std = std_sum / total_samples

    return overall_mean, overall_std


# 图像变换
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 设置路径
DATA_DIR = r'/data1/ybw/3D/ybw_data_3cam/my_picture_flow_feed/merge_data_all'
csv_file_path = os.path.join(DATA_DIR, 'data_analysis.csv')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is used:{device}')

# 计算每个角度的均值和标准差
angles = ['45', '90', 'besides']
batch_size = 128
num_workers = 8

for angle in angles:
    print(f"\nCalculating for {angle} degree images:")

    dataset = MyDataset(csv_file=csv_file_path, root_dir=DATA_DIR, angle=angle, transform=transform)
    print(f"Total valid images for {angle}: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean, std = calculate_mean_std(dataloader, device)

    print(f"{angle} degree Mean:", mean)
    print(f"{angle} degree Std:", std)

"""

merge-data-radom:
    if angle == 45:
        MEAN = [0.3452, 0.3804, 0.4261]
        STD = [0.2656, 0.2769, 0.3000]
    elif angle == 90:
        MEAN = [0.4124, 0.4363, 0.4868]
        STD = [0.2735, 0.2696, 0.2823]
    else:  # besides
        MEAN = [0.4002, 0.4100, 0.3871]
        STD = [0.2855, 0.2829, 0.2926]
        
merge-data-fixed:
    if angle == 45:
        MEAN = [0.3148, 0.3729, 0.4246]
        STD = [0.2727, 0.2826, 0.3225]
    elif angle == 90:
        MEAN = [0.3067, 0.3332, 0.3712]
        STD = [0.2338, 0.2435, 0.2837]
    else:  # besides
        MEAN = [0.3332, 0.3531, 0.3221]
        STD = [0.2707, 0.2714, 0.2839]

merge-data-all:
    if angle == 45:
        MEAN = [0.3260, 0.3757, 0.4252]
        STD = [0.2702, 0.2806, 0.3142]
    elif angle == 90:
        MEAN = [0.3457, 0.3713, 0.4139]
        STD = [0.2488, 0.2535, 0.2835]
    else:  # besides
        MEAN = [0.3579, 0.3742, 0.3461]
        STD = [0.2763, 0.2757, 0.2873]
"""