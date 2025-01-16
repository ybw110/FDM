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


class RelationAttention(nn.Module):
    def __init__(self, fea_out):
        super(RelationAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(fea_out, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, relations):
        weights = [self.attention(rel) for rel in relations]
        weights = torch.softmax(torch.cat(weights, dim=1), dim=1)
        weighted_sum = sum(w * r for w, r in zip(weights.split(1, dim=1), relations))
        return weighted_sum


class RelationBlock_Out(nn.Module):
    def __init__(self, fea_out):
        super(RelationBlock_Out, self).__init__()
        self.linear_out = nn.Sequential(
            nn.Linear(fea_out * fea_out, fea_out),
            nn.BatchNorm1d(fea_out),
            nn.ReLU(inplace=True)
        )
        self.relation_attention = RelationAttention(fea_out)

    def cal_relation(self, input1, input2):
        input1 = input1.unsqueeze(2)
        input2 = input2.unsqueeze(1)
        outproduct = torch.bmm(input1, input2)
        return outproduct

    def forward(self, x):
        relation_eachview_list = []
        for i in range(len(x)):
            relation_list = []
            for j in range(len(x)):
                if i != j:
                    relation_temp = self.cal_relation(x[i], x[j])
                    relation_temp = relation_temp.view(relation_temp.size(0), -1)
                    relation_temp = self.linear_out(relation_temp)
                    relation_list.append(relation_temp)
            relation_eachview_temp = self.relation_attention(relation_list)
            relation_eachview_list.append(relation_eachview_temp)
        return relation_eachview_list


class MultiViewNet(nn.Module):
    def __init__(self, num_classes=5, num_views=3, fea_out=512, fea_com=512):
        super(MultiViewNet, self).__init__()
        self.num_views = num_views
        self.fea_out = fea_out
        self.fea_com = fea_com

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

        self.view_attention = ViewAttention(fea_out)
        self.relation_out = RelationBlock_Out(fea_out)

        self.classifier_out = nn.Sequential(
            nn.Linear(fea_out * 2, fea_com),
            nn.BatchNorm1d(fea_com),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fea_com, num_classes),
            nn.BatchNorm1d(num_classes)
        )

    def forward(self, input):
        # Extract features and apply self-attention
        Fea_list = [self.self_attentions[i](self.view_models[i](input_item))
                    for i, input_item in enumerate(input)]

        # Apply view-level attention
        attended_fea = self.view_attention(Fea_list)

        # Calculate relations with attention
        Relation_fea = self.relation_out(Fea_list)

        # Combine features and relations for each view
        Fea_Relation_list = []
        for k in range(len(Fea_list)):
            Fea_Relation_temp = torch.cat((Fea_list[k], Relation_fea[k]), 1)
            Fea_Relation_list.append(self.classifier_out(Fea_Relation_temp))

        return Fea_Relation_list, attended_fea

