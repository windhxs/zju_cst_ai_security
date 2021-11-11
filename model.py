from time import sleep
import torch 
import torch.nn as nn
from torch.nn.modules.pooling import AdaptiveAvgPool2d


def conv3x3(inplanes, planes, padding, stride=1,):
    return nn.Conv2d(inplanes, planes, 3, stride=stride, padding=padding)

def conv1x1(inplanes, planes, padding, stride=1,):
    return nn.Conv2d(inplanes, planes, 1, stride=stride, padding=padding)


class Resnet_block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, padding=1, stride=stride)
        self.norm1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, padding=1, stride=stride)
        self.norm2 = nn.BatchNorm2d(planes)
        self.shorcut = conv1x1(inplanes, planes, padding=0)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.conv2(out)
        out = self.norm2(out)

        identity = self.shorcut(identity)
        # print(out.shape, identity.shape)
        out = self.relu(out + identity)
        return out   
        

class CIFAR10Model(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.planes = 128
        self.dilation = 1
        self.norm = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, self.planes, kernel_size=3, stride=2, padding=3, bias=False)
        self.bn1 = self.norm(self.planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(128, 2)
        self.layer2 = self._make_layer(256, 2)
        self.layer3 = self._make_layer(512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def forward(self, x):
        out = self.conv1(x)
        # print(out.shape)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # print(out.shape)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # print(out.shape)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def _init_weights(self):
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

        

# import torch
# import torch.nn as nn


# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }


# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Linear(512, 10)

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         return out

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)
