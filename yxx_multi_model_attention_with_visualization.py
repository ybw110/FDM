import torch
import torch.nn as nn
import torchvision.models as models

class ViewAttention(nn.Module):
    def __init__(self, fea_out):
        super(ViewAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(fea_out, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        weights = [self.attention(feature) for feature in x]
        weights = torch.softmax(torch.cat(weights, dim=1), dim=1)
        weighted_sum = sum(w * f for w, f in zip(weights.split(1, dim=1), x))
        return weighted_sum


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
    def __init__(self, num_classes=5, num_views=3, fea_out=512, fea_com=512):
        super(MultiViewNet, self).__init__()
        self.num_views = num_views
        self.fea_out = fea_out
        self.fea_com = fea_com

        # 创建多个视图模型
        self.view_models = nn.ModuleList()
        self.self_attentions = nn.ModuleList()
        for i in range(num_views):
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, fea_out),
                nn.BatchNorm1d(fea_out),
                nn.ReLU(inplace=True),
                nn.Dropout()
            )
            self.view_models.append(model)
            self.self_attentions.append(SelfAttention(fea_out))

        # 创建视图注意力层
        self.view_attention = ViewAttention(fea_out)

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
        ResNet_outputs = []
        SelfAttention_outputs = []
        Fea_list = []

        # 提取特征并应用自注意力机制
        for i, input_item in enumerate(input):
            # 从每个视图模型获取特征
            resnet_output = self.view_models[i](input_item)
            ResNet_outputs.append(resnet_output)

            # 应用自注意力机制
            self_attention_output = self.self_attentions[i](resnet_output)
            Fea_list.append(self_attention_output)
            SelfAttention_outputs.append(self_attention_output)

        # 视图级别的注意力机制
        ViewAttention_output = self.view_attention(Fea_list)

        # 使用融合特征进行分类
        final_classification = self.classifier_out(ViewAttention_output)

        # 返回分类结果以及中间特征图
        return final_classification, ResNet_outputs, SelfAttention_outputs, ViewAttention_output