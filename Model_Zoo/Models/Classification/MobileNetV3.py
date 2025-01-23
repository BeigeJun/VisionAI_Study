import torch.nn as nn

from Model_Zoo.Models.Classification.Convolution_Modules import InvertedResidualBlock
from Model_Zoo.Models.Classification.Convolution_Modules import _make_divisible
from Model_Zoo.Models.Classification.Convolution_Modules import h_swish
from Model_Zoo.Models.Model_Base.ModelBase import modelbase


class MobileNetV3(modelbase):
    def __init__(self, alpha=1.0, num_class=10):
        super().__init__()
        large = [
            # Input, Output, Expansion, Kernel, Stride, SE, UseHS
            [16, 16, 1, 3, 1, False, False],
            [16, 24, 4, 3, 2, False, False],
            [24, 24, 3, 3, 1, False, False],
            [24, 40, 3, 5, 2, False, True],
            [40, 40, 3, 5, 1, False, True],
            [40, 40, 3, 5, 1, False, True],
            [40, 80, 6, 3, 2, True, False],
            [80, 80, 2.5, 3, 1, True, False],
            [80, 80, 2.4, 3, 1, True, False],
            [80, 80, 2.4, 3, 1, True, False],
            [80, 112, 6, 3, 1, True, True],
            [112, 112, 6, 3, 1, True, True],
            [112, 160, 6, 5, 2, True, True],
            [160, 160, 6, 5, 1, True, True],
            [160, 160, 6, 5, 1, True, True]
        ]
        small = [
            [16, 16, 1, 3, 2, False, True],
            [16, 24, 4, 3, 2, False, False],
            [24, 24, 11.0 / 3.0, 3, 1, False, False],
            [24, 40, 4, 5, 2, True, True],
            [40, 40, 6, 5, 1, True, True],
            [40, 40, 6, 5, 1, True, True],
            [40, 48, 3, 5, 1, True, True],
            [48, 48, 3, 5, 1, True, True],
            [48, 96, 6, 5, 2, True, True],
            [96, 96, 6, 5, 1, True, True],
            [96, 96, 6, 5, 1, True, True],
        ]

        in_channel = _make_divisible(16 * alpha)
        out_channel = 0
        self.First_Step = nn.Sequential(
            nn.Conv2d(3, in_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(16*alpha)),
            nn.ReLU6(inplace=True)
        )
####################################################################################################################
        layers = []
        for i in range(len(large)):
            out_channel = _make_divisible(large[i][1] * alpha)
            layers.append(InvertedResidualBlock(in_channel, out_channel,
                                                expansion=large[i][2], kernel=large[i][3], stride=large[i][4],
                                                se=large[i][5], re=large[i][6]))
            in_channel = out_channel

        self.Second_Step = nn.Sequential(*layers)

        self.Third_Step = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * 6, 1, 1),
            nn.BatchNorm2d(out_channel * 6),
            h_swish(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(960, 1280),
            nn.BatchNorm2d(1280),
            h_swish(),
            nn.Linear(1280, num_class)
            )

    def forward(self, x):
        x = self.First_Step(x)
        x = self.Second_Step(x)
        x = self.Third_Step(x)
        return x
