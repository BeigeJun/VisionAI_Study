import torch.nn as nn

from F_Model_Zoo.Models.Classification.Convolution_Modules import InvertedResidualBlock
from F_Model_Zoo.Models.Util.ModelBase import modelbase


class MobileNetV2(modelbase):
    def __init__(self, alpha=1.0, num_class=10):
        super().__init__()
        configs = [
            # Input, Output, Expansion, kernel, Stride, Repeat
            [32, 16, 6, 3, 1, 1],
            [16, 24, 6, 3, 2, 2],
            [24, 32, 6, 3, 1, 3],
            [32, 64, 6, 3, 2, 4],
            [64, 96, 6, 3, 1, 3],
            [96, 160, 6, 3, 1, 3],
            [160, 320, 6, 3, 2, 1]
        ]
        self.First_Step = nn.Sequential(
            nn.Conv2d(3, int(32*alpha), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(32*alpha)),
            nn.ReLU6(inplace=True)
        )

        layers = []
        for i in range(len(configs)):
            layers.append(InvertedResidualBlock(int(configs[i][0] * alpha), int(configs[i][1] * alpha),
                                                expansion=configs[i][2], kernel=configs[i][3], stride=configs[i][4]))
            for _ in range(configs[i][4]-1):
                layers.append(InvertedResidualBlock(int(configs[i][1]*alpha), int(configs[i][1]*alpha),
                                                    expansion=configs[i][2], kernel=configs[i][3],
                                                    stride=configs[i][4]))

        self.Second_Step = nn.Sequential(*layers)
        self.Third_Step = nn.Sequential(
            nn.Conv2d(int(320*alpha), int(1280*alpha), 1, 1),
            nn.BatchNorm2d(int(1280*alpha)),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())
        self.Last_Conv = nn.Linear(int(1280*alpha), num_class)

    def forward(self, x):
        x = self.First_Step(x)
        x = self.Second_Step(x)
        x = self.Third_Step(x)
        x = self.Last_Conv(x)
        return x
