import torch
from torch import nn
from torchvision import models

from src.deep_clusterers.models import resnet


class DeepClusterer(nn.Module):
    def __init__(self, backbone, last_channel=1280, num_classes=None, **kwargs):
        super(DeepClusterer, self).__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_channel, num_classes)

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


def MobileNet(**kwargs):
    model = models.mobilenet_v2(**kwargs)
    last_channel = 1280
    return DeepClusterer(model, last_channel=last_channel, **kwargs)


def ResNet18(**kwargs):
    model = resnet.resnet18(**kwargs)
    last_channel = 512
    return DeepClusterer(model, last_channel=last_channel, **kwargs)


def ResNet34(**kwargs):
    model = resnet.resnet34(**kwargs)
    last_channel = 512
    return DeepClusterer(model, last_channel=last_channel, **kwargs)


def ResNet50(**kwargs):
    model = resnet.resnet50(**kwargs)
    last_channel = 2048
    return DeepClusterer(model, last_channel=last_channel, **kwargs)


def ResNet101(**kwargs):
    model = resnet.resnet101(**kwargs)
    last_channel = 2048
    return DeepClusterer(model, last_channel=last_channel, **kwargs)


def ResNet152(**kwargs):
    model = resnet.resnet152(**kwargs)
    last_channel = 2048
    return DeepClusterer(model, last_channel=last_channel, **kwargs)


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