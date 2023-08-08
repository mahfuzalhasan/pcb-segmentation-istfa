import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from .resnet import ResNet_BasicBlock_OS16, ResNet_BasicBlock_OS8
from .aspp import ASPP

class DeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3, self).__init__()

        self.num_classes = num_classes

        # self.model_id = model_id
        # self.project_dir = project_dir
        # self.create_model_dirs()

        self.resnet = ResNet_BasicBlock_OS8(num_layers=18) # NOTE! specify the type of ResNet here
        self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return output


if __name__=="__main__":
    x = torch.randn(8, 3, 512, 512)
    model = DeepLabV3(2)
    y = model(x)

    print(y.size())
