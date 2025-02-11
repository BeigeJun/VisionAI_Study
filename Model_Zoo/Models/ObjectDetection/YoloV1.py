import torch.nn as nn
from torchvision import transforms
from Model_Zoo.Models.Util.ModelBase import modelbase

class Yolov1(modelbase):
    def __init__(self, split_size, num_boxes, num_classes=20):
        super().__init__()

        self.transform_info = [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ]

        self.cfg = [
            # Input, Output, Kernel, Stride, Pool
            [3, 64, 7, 2, True],
            [64, 192, 3, 1, True],
            [192, 128, 1, 1, False],
            [128, 256, 3, 1, False],
            [256, 256, 1, 1, False],
            [256, 512, 3, 1, True],
            [512, 256, 1, 1, False],
            [256, 512, 3, 1, False],
            [512, 256, 1, 1, False],
            [256, 512, 3, 1, False],
            [512, 256, 1, 1, False],
            [256, 512, 3, 1, False],
            [512, 256, 1, 1, False],
            [256, 512, 3, 1, False],
            [512, 512, 1, 1, False],
            [512, 1024, 3, 1, True],
            [1024, 512, 1, 1, False],
            [512, 1024, 3, 1, False],
            [1024, 512, 1, 1, False],
            [512, 1024, 3, 1, False],
            [1024, 1024, 3, 1, False],
            [1024, 1024, 3, 2, False],
            [1024, 1024, 3, 1, False],
            [1024, 1024, 3, 1, False]
        ]
        layers = []
        for Input, Output, Kernel, Stride, Pool in self.cfg:
            layers.append(nn.Conv2d(in_channels=Input, out_channels=Output,
                                    kernel_size=Kernel, stride=Stride, padding=Kernel//2))
            layers.append(nn.BatchNorm2d(Output))
            layers.append(nn.LeakyReLU(0.1))
            if Pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.First_layer = nn.Sequential(*layers)
        self.Second_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * split_size * split_size, 496),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            #마지막 7*7*30의 아웃풋이 7*7로 나눠진 영역의 바운딩박스 2개 * (x, y, w, h, confidence) + 각 클래스의 확률
            nn.Linear(496, split_size * split_size * (5 * num_boxes + num_classes))
        )

    def forward(self, x):
        out = self.First_layer(x)
        return self.Second_layer(out)

    def return_transform_info(self):
        return self.transform_info
