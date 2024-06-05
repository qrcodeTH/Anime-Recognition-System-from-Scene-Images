import torch.nn as nn

class MyMobileNetV2(nn.Module):
    def __init__(self, num_classes=100):
        super(MyMobileNetV2, self).__init__()
        self.network = models.mobilenet_v2(pretrained=True)

        in_features = self.network.classifier[1].in_features
        self.network.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.network(x)
