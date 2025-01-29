import torch.nn as nn
from torchvision import transforms
from Model_Zoo.Models.Classification.Convolution_Modules import ResidualBlock
from Model_Zoo.Models.Util.ModelBase import modelbase


class ResNet(modelbase):
    def __init__(self, model_type='50', num_class=10):
        super().__init__()

        self.transform_info = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.expansion = 4

        self.resnet50 = [
            #Out, Stride, Repeat
            [64, 1, 3],
            [128, 2, 4],
            [256, 2, 6],
            [512, 2, 3]
        ]
        self.resnet101 = [
            [64, 1, 3],
            [128, 2, 4],
            [256, 2, 23],
            [512, 2, 3]
        ]
        self.resnet152 = [
            [64, 1, 3],
            [128, 2, 8],
            [256, 2, 36],
            [512, 2, 3]
        ]
        if model_type == '50':
            self.model = self.resnet50
        elif model_type == '101':
            self.model = self.resnet101
        else:
            self.model = self.resnet152

        self.First_step = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU6()
        )

        self.Max_Pool = nn.MaxPool2d(3, 2, 1)

        layer = []
        prev_channels = 64

        for i in range(len(self.model)):
            in_channels = prev_channels
            out_channels = self.model[i][0]

            for j in range(self.model[i][2]):
                layer.append(ResidualBlock(in_channels, out_channels,
                                           stride=self.model[i][1] if j == 0 else 1,
                                           expansion=self.expansion))
                in_channels = out_channels * self.expansion
            prev_channels = out_channels * self.expansion

        self.Second_step = nn.Sequential(*layer)

        self.Third_step = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_class)
        )

    def forward(self, x):
        x = self.First_step(x)
        x = self.Max_Pool(x)
        x = self.Second_step(x)
        x = self.Third_step(x)
        return x

    def return_transform_info(self):
        return self.transform_info
