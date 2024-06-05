import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torchvision.models as models

class MyEfficientNetB0(nn.Module):
    def __init__(self):
        super(MyEfficientNetB0, self).__init__()
        self.network = EfficientNet.from_pretrained("efficientnet-b0")
        self.network._fc = nn.Linear(self.network._fc.in_features, 512)
        self.classifier = nn.Linear(512, 100)

    def forward(self, x):
        out = self.network(x)
        return out

class MyEfficientNetB7(nn.Module):
    def __init__(self):
        super(MyEfficientNetB7, self).__init__()
        self.network = EfficientNet.from_pretrained("efficientnet-b7")
        self.network._fc = nn.Linear(self.network._fc.in_features, 512)
        self.classifier = nn.Linear(512, 100)

    def forward(self, x):
        out = self.network(x)
        return out

class MyResNeXt101(nn.Module):
    def __init__(self, num_classes=100):
        super(MyResNeXt101, self).__init__()
        self.network = models.resnext101_32x8d(pretrained=True)
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.network(x)
