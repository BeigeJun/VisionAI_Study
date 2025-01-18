import os
import sys
import torch.nn as nn

from Convolution_Modules import DepthWiseSeparableConv
from Model_Zoo.Models.Model_Base.ModelBase import  modelbase

class MobileNetV1(modelbase):
    def __init__(self, alpa=1.0, num_class=10):
        super().__init__() #부모 클래스 초기화
        self.alpa = alpa
        self.configs = [
            # c, n, s
            [32, 1, 2],
            [24, 2, 2],
            [32, 3, 2],
        ]

    def forward(self, x):
        x = nn.Conv2d(3, int(32 * alpha), 3, stride=2, padding=1, bias=False)(x)
        nn.BatchNorm2d(int(32 * alpha)),
        nn.ReLU(inplace=True),