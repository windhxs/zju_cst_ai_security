import torch 
import torch.nn as nn


def conv3x3(inplanes, planes, stride=1):
    return nn.Conv2d(inplanes, planes, 3, stride=stride)


class Resnet_block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, ) -> None:
        super().__init__()


        self.conv1 =  conv3x3(inplanes, planes, stride)
        self.norm1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

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
    def __init__(self) -> None:
        super().__init__()

        self.planes = 32

    def _make_layer(planes, stride=1):
        


        