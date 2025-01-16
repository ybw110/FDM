import torch
import torch.nn as nn
import torchvision.models as models



# 不同论文方法的模型
import torch
import torch.nn as nn
import torchvision.models as models

# Jin 等 (2019)
class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=5):  # 将类别数量改为5
        super(ModifiedResNet50, self).__init__()
        # 加载预训练的 ResNet50 模型
        self.resnet50 = models.resnet50(pretrained=True)
        # 获取 ResNet50 最后一层全连接层的输入特征维数
        num_ftrs = self.resnet50.fc.in_features
        # 替换 ResNet50 的最后一层全连接层，输出5类别
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),  # 添加一个全连接层
            nn.ReLU(),  # 使用 ReLU 激活函数
            nn.Linear(1024, num_classes)  # 最终输出 num_classes 个类别
        )

    def forward(self, x):
        return self.resnet50(x)


# Brion 和 Pattinson (2022) - 多头残差注意力网络
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

