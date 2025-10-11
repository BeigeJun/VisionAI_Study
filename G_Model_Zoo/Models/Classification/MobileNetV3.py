import torch.nn as nn
from torchvision import transforms
from G_Model_Zoo.Models.Classification.Convolution_Modules import InvertedResidualBlock
from G_Model_Zoo.Models.Classification.Convolution_Modules import _make_divisible
from G_Model_Zoo.Models.Classification.Convolution_Modules import h_swish
from G_Model_Zoo.Models.Util.ModelBase import modelbase


class MobileNetV3(modelbase):
    def __init__(self, model_type='large', alpha=1.0, num_class=10):
        super().__init__()

        self.transform_info = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

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

        if model_type == 'small':
            model = small
            last_linear = 1024
        elif model_type == 'large':
            model = large
            last_linear = 1280
        else:
            model = None
            last_linear = 0

        in_channel = _make_divisible(16 * alpha)
        out_channel = 0
        self.First_Step = nn.Sequential(
            nn.Conv2d(3, in_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(16*alpha)),
            nn.ReLU6(inplace=True)
        )
        layers = []
        for i in range(len(model)):
            out_channel = _make_divisible(model[i][1] * alpha)
            layers.append(InvertedResidualBlock(in_channel, out_channel,
                                                expansion=model[i][2], kernel=model[i][3], stride=model[i][4],
                                                se=model[i][5], re=model[i][6]))
            in_channel = out_channel

        self.Second_Step = nn.Sequential(*layers)

        self.Third_Step = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * 6, 1, 1),
            nn.BatchNorm2d(out_channel * 6),
            h_swish(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channel * 6, last_linear),
            h_swish(),
            nn.Linear(last_linear, num_class)
            )

    def forward(self, x):
        x = self.First_Step(x)
        x = self.Second_Step(x)
        x = self.Third_Step(x)
        return x

    def return_transform_info(self):
        return self.transform_info
