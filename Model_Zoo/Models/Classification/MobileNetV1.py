import torch.nn as nn

from Model_Zoo.Models.Classification.Convolution_Modules import DepthWiseSeparableConv
from Model_Zoo.Models.Model_Base.ModelBase import modelbase


class MobileNetV1(modelbase):
    def __init__(self, alpha=1.0, num_class=10):
        super().__init__()
        self.configs = [
            # Input, Output, Stride, Repeat
            [32, 64, 1, 1],
            [64, 128, 2, 1],
            [128, 128, 1, 1],
            [128, 256, 2, 1],
            [256, 256, 1, 1],
            [256, 512, 2, 1],
            [512, 512, 1, 6],
            [512, 1024, 2, 1],
            [1024, 1024, 1, 1]
        ]
        self.First_Step = nn.Sequential(
            nn.Conv2d(3, int(32*alpha), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(32*alpha)),
            nn.ReLU(inplace=True)
        )

        layers = []
        for Input, OutPut, Stride, Repeat in self.configs:
            for _ in range(Repeat):
                layers.append(DepthWiseSeparableConv(int(Input*alpha), int(OutPut*alpha), stride=Stride))

        self.Second_Step = nn.Sequential(*layers)
        self.Third_Step = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(int(1024*alpha), num_class)
        )

    def forward(self, x):
        x = self.First_Step(x)
        x = self.Second_Step(x)
        x = self.Third_Step(x)
        return x
