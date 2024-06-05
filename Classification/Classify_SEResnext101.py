import torch.nn as nn

class MySEResNeXt101(nn.Module):
    def __init__(self, num_classes=100):
        super(MySEResNeXt101, self).__init__()
        self.network = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d', pretrained=True)

        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.network(x)
