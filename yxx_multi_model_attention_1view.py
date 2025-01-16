import torch
import torch.nn as nn
import torchvision.models as models

class SelfAttention(nn.Module):
    def __init__(self, fea_out):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(fea_out, fea_out)
        self.key = nn.Linear(fea_out, fea_out)
        self.value = nn.Linear(fea_out, fea_out)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention = torch.softmax(torch.bmm(q.unsqueeze(1), k.unsqueeze(2)) / (x.size(-1) ** 0.5), dim=2)
        out = torch.bmm(attention, v.unsqueeze(1)).squeeze(1)
        return out


class MultiViewNet(nn.Module):
    def __init__(self, num_classes=5, fea_out=512, fea_com=512):
        super(MultiViewNet, self).__init__()
        self.fea_out = fea_out
        self.fea_com = fea_com

        # 创建单个视图模型
        self.view_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.view_model.fc = nn.Sequential(
            nn.Linear(self.view_model.fc.in_features, fea_out),
            nn.BatchNorm1d(fea_out),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        # 自注意力层
        self.self_attention = SelfAttention(fea_out)

        # 分类器
        self.classifier_out = nn.Sequential(
            nn.Linear(fea_out, fea_com),
            nn.BatchNorm1d(fea_com),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fea_com, num_classes),
            nn.BatchNorm1d(num_classes)
        )

    def forward(self, input):
        # 确保输入是 4D 的
        if input.dim() == 3:
            input = input.unsqueeze(0)  # 添加 batch 维度

        # 通过单视角模型和自注意力机制
        fea = self.view_model(input)
        fea = self.self_attention(fea)

        # 使用提取到的特征进行分类
        final_classification = self.classifier_out(fea)

        return final_classification, fea