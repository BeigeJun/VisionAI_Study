import math
import torch.nn as nn
from torchvision import transforms
from Model_Zoo.Models.Classification.Convolution_Modules import _make_divisible
from Model_Zoo.Models.Classification.Convolution_Modules import MBconv
from Model_Zoo.Models.Util.ModelBase import modelbase


class EfficientNet(modelbase):
    def __init__(self, depth_mult=1.0, width_mult=1.0, resize=1.0, crop=1.0, drop=0.2, num_class=10):
        super().__init__()

        self.transform_info = transforms.Compose([
            transforms.Resize((int(224*resize), int(224*resize))),
            transforms.CenterCrop(int(224*resize*crop)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.configs = [
            # Output, Expansion, Kernel, Stride, Repeat
            [16, 1, 3, 1, 1],
            [24, 6, 3, 2, 2],
            [40, 6, 5, 2, 2],
            [80, 6, 3, 1, 3],
            [112, 6, 5, 2, 3],
            [192, 6, 5, 2, 4],
            [320, 6, 3, 1, 1]
        ]
        in_put = _make_divisible(32 * width_mult, divisor=8)
        self.First_step = nn.Sequential(
            nn.Conv2d(3, in_put, 3, 2, 1),
            nn.BatchNorm2d(in_put),
            nn.SiLU(inplace=True)
        )
        layers = []
        for Output, Expansion, Kernel, Stride, Repeat in self.configs:
            repeat = math.ceil(Repeat*depth_mult)#깊이 조정
            for i in range(repeat):
                stride = Stride if i == 0 else 1 #repeat만큼 반복할 때 처음만 변경사항. 나머진 1로 고정
                output = _make_divisible(Output*width_mult, 8) #너비 조정
                expansion_channel = _make_divisible(in_put * Expansion, 8)
                layers.append(MBconv(in_put, output, Kernel, stride, expansion_channel))
                in_put = output

        self.Second_step = nn.Sequential(*layers)

        last_output = _make_divisible(int(1280 * width_mult), 8)
        self.Third_step = nn.Sequential(
            nn.Conv2d(in_put, last_output, 1),
            nn.BatchNorm2d(last_output),
            nn.SiLU(inplace=True)
        )
        self.Fourth_step = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(last_output, num_class)
        )

    def forward(self, x):
        out = self.First_step(x)
        out = self.Second_step(out)
        out = self.Third_step(out)
        out = self.Fourth_step(out)
        return out

    def return_transform_info(self):
        return self.transform_info
