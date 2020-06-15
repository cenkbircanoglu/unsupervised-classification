import math

import torch
from torch import nn
from torchvision import models

from src.deep_clusterers.models import resnet


class DeepClusterer(nn.Module):
    def __init__(self, backbone, last_channel=1280, num_classes=None, initialize=False, **kwargs):
        super(DeepClusterer, self).__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_channel, num_classes)
        if initialize:
            self._initialize_weights()

    def reinitialize_fc(self):
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def extract_features(self, x):
        x = self.backbone.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


def MobileNet(initialize=None, **kwargs):
    model = models.mobilenet_v2(**kwargs)
    last_channel = 1280
    return DeepClusterer(model, last_channel=last_channel, initialize=initialize, **kwargs)


def ResNet18(initialize=None, **kwargs):
    model = resnet.resnet18(**kwargs)
    last_channel = 512
    return DeepClusterer(model, last_channel=last_channel, initialize=initialize, **kwargs)


def ResNet34(initialize=None, **kwargs):
    model = resnet.resnet34(**kwargs)
    last_channel = 512
    return DeepClusterer(model, last_channel=last_channel, initialize=initialize, **kwargs)


def ResNet50(initialize=None, **kwargs):
    model = resnet.resnet50(**kwargs)
    last_channel = 2048
    return DeepClusterer(model, last_channel=last_channel, initialize=initialize, **kwargs)


def ResNet101(initialize=None, **kwargs):
    model = resnet.resnet101(**kwargs)
    last_channel = 2048
    return DeepClusterer(model, last_channel=last_channel, initialize=initialize, **kwargs)


def ResNet152(initialize=None, **kwargs):
    model = resnet.resnet152(**kwargs)
    last_channel = 2048
    return DeepClusterer(model, last_channel=last_channel, initialize=initialize, **kwargs)


if __name__ == "__main__":
    input1 = torch.autograd.Variable(torch.randn(2, 3, 32, 32))
    input2 = torch.autograd.Variable(torch.randn(2, 3, 256, 256))

    model = MobileNet(num_classes=100)
    o1 = model(input1)
    print(o1.shape, input1.shape)
    o2 = model(input2)
    print(o2.shape, input2.shape)

    model = ResNet18(num_classes=100)
    o1 = model(input1)
    print(o1.shape, input1.shape)
    o2 = model(input2)
    print(o2.shape, input2.shape)

    model = ResNet34(num_classes=100)
    o1 = model(input1)
    print(o1.shape, input1.shape)
    o2 = model(input2)
    print(o2.shape, input2.shape)

    model = ResNet50(num_classes=100)
    o1 = model(input1)
    print(o1.shape, input1.shape)
    o2 = model(input2)
    print(o2.shape, input2.shape)

    model = ResNet101(num_classes=100)
    o1 = model(input1)
    print(o1.shape, input1.shape)
    o2 = model(input2)
    print(o2.shape, input2.shape)

    model = ResNet152(num_classes=100)
    o1 = model(input1)
    print(o1.shape, input1.shape)
    o2 = model(input2)
    print(o2.shape, input2.shape)
