from time import sleep
import torch 
import torch.nn as nn
from torch.nn.modules.pooling import AdaptiveAvgPool2d


def conv3x3(inplanes, planes, stride=1):
    return nn.Conv2d(inplanes, planes, 3, stride=stride)


class Resnet_block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, ) -> None:
        super().__init__()
        self.conv1 =  conv3x3(inplanes, planes, stride)
        self.norm1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, stride)
        self.norm2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out + identity)
        return out   
        

class CIFAR10Model(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.planes = 32
        self.dilation = 1
        self.norm = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, self.planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm(self.planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(32, 2)
        self.layer2 = self._make_layer(64, 2)
        self.avgpool = nn,AdaptiveAvgPool2d(1,1)
        self.fc = nn.Linear()

        self._init_weights()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_block, stride=1):
        norm = self.norm
        dilation = self.dilation

        layers = []
        layers.append(Resnet_block(self.planes, planes, stride=stride))

        self.planes = planes

        for _ in range(1, num_block):
            layers.append(Resnet_block(self.planes, planes, stride=stride))
        return nn.Sequential(*layers)

        