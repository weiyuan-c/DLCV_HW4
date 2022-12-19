import torch.nn as nn
from torch import Tensor


class ResNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer2(self.layer1(x))
        x = self.layer4(self.layer3(x))
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

class Classifier(nn.Module):
    def __init__(self, model, freeze=False):
        super().__init__()
        self.backbone = ResNet(model)
        if freeze:
            for name, child in self.backbone.named_children():
                for param in child.parameters():
                    param.requires_grad = False
            print('backbone is fixed!')
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 65)
        )
    def forward(self, x: Tensor):
        x = self.backbone(x)
        return self.fc(x)