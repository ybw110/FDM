import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchmetrics import Accuracy, Precision, Recall, F1Score
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar


# ===================== Dataset and DataLoader =====================

class ParametersDataset(Dataset):
    def __init__(self, csv_file, root_dir, angle, transform=None, image_dim=(320, 320)):
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.angle = angle
        self.transform = transform
        self.image_dim = image_dim

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
            raise FileNotFoundError(f"File not found: {img_name}")

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

        old_label = flow_rate_class * 3 + feed_rate_class
        if old_label == 2:
            new_label = 0  # Severe under-extrusion
        elif old_label in [0, 1, 5]:
            new_label = 1  # Mild under-extrusion
        elif old_label == 4:
            new_label = 2  # Normal
        elif old_label in [3, 7, 8]:
            new_label = 3  # Mild over-extrusion
        elif old_label == 6:
            new_label = 4  # Severe over-extrusion
        else:
            raise ValueError(f"Invalid label combination: {old_label}")

        label = torch.zeros(5)
        label[new_label] = 1

        return image, label


class ParametersDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, root_dir, angle, batch_size=32, num_workers=4):
        super().__init__()
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.angle = angle
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Creating dataset with transformations
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3260, 0.3757, 0.4252], [0.2702, 0.2806, 0.3142])
        ])
        full_dataset = ParametersDataset(self.csv_file, self.root_dir, self.angle, transform=transform)

        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset,
                                                                               [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# ===================== Model Definition =====================

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return nn.ReLU()(out)
# 定义 Attention 模块
import torch.nn.functional as F
class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionModule, self).__init__()
        self.trunk = ResidualBlock(in_channels, out_channels)
        self.mask = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        trunk_output = self.trunk(x)
        mask_output = self.mask(x)
        # 上采样 mask_output 以匹配 trunk_output 的尺寸
        if mask_output.size() != trunk_output.size():
            mask_output = F.interpolate(mask_output, size=trunk_output.shape[2:], mode='bilinear', align_corners=False)
        out = trunk_output * mask_output
        return out

# 定义多头残差注意力网络（单头版本，5分类）
class MultiHeadAttentionNetwork(nn.Module):
    def __init__(self, num_classes=5):  # 将类别数设置为5
        super(MultiHeadAttentionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # 输入为RGB图像
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = ResidualBlock(64, 128)
        self.attention1 = AttentionModule(128, 256)
        self.attention2 = AttentionModule(256, 512)
        self.attention3 = AttentionModule(512, 512)

        self.fc = nn.Linear(512 * 112 * 112, num_classes)  # 全连接层，输出5个类别

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.attention1(x)
        x = self.attention2(x)
        x = self.attention3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x



class ParametersClassifier(pl.LightningModule):
    def __init__(self, num_classes=5, lr=1e-3, weight_decay=1e-5):
        super(ParametersClassifier, self).__init__()
        self.model = MultiHeadAttentionNetwork(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay

        # Metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.train_precision = Precision(num_classes=num_classes, average='weighted')
        self.val_precision = Precision(num_classes=num_classes, average='weighted')
        self.train_recall = Recall(num_classes=num_classes, average='weighted')
        self.val_recall = Recall(num_classes=num_classes, average='weighted')
        self.train_f1 = F1Score(num_classes=num_classes, average='weighted')
        self.val_f1 = F1Score(num_classes=num_classes, average='weighted')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels.argmax(dim=1))

        # Log metrics
        preds = torch.argmax(outputs, dim=1)
        self.train_acc(preds, labels.argmax(dim=1))
        self.train_precision(preds, labels.argmax(dim=1))
        self.train_recall(preds, labels.argmax(dim=1))
        self.train_f1(preds, labels.argmax(dim=1))

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        self.log('train_precision', self.train_precision, prog_bar=True)
        self.log('train_recall', self.train_recall, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels.argmax(dim=1))

        # Log metrics
        preds = torch.argmax(outputs, dim=1)
        self.val_acc(preds, labels.argmax(dim=1))
        self.val_precision(preds, labels.argmax(dim=1))
        self.val_recall(preds, labels.argmax(dim=1))
        self.val_f1(preds, labels.argmax(dim=1))

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_precision', self.val_precision, prog_bar=True)
        self.log('val_recall', self.val_recall, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


# ===================== Training Script =====================

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1234, type=int, help="Set seed for training")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train the model for")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    return parser.parse_args()


def main():
    args = args_parser()

    # Data module
    data_module = ParametersDataModule(
        csv_file="path/to/csv",
        root_dir="path/to/data",
        angle="45",  # or "90" or "besides"
        batch_size=args.batch_size,
        num_workers=4
    )

    # Model
    model = ParametersClassifier(num_classes=5, lr=args.lr)

    # Loggers
    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    csv_logger = pl_loggers.CSVLogger("logs/", name="my_model_logs")

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min"
    )

    # Progress bar
    progress_bar = TQDMProgressBar()

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_callback, progress_bar],
        precision=16,  # Mixed precision training
        gpus=1 if torch.cuda.is_available() else 0
    )

    # Start training
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()