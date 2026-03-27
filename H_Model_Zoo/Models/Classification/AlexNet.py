import torch.nn as nn
from torchvision import transforms
from G_Model_Zoo.Models.Classification.Convolution_Modules import conv_n_layer_block
from G_Model_Zoo.Models.Util.ModelBase import modelbase


class AlexNet(modelbase):
    def __init__(self, dim=64, num_class=10):
        super().__init__()

        self.transform_info = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        self.cfg = [
            #Input, Output, Kernel, Padding, MaxPool
            [3, dim, 5, 2, True],
            [dim, dim * 3, 5, 2, True],
            [dim * 3, dim * 6, 3, 1, False],
            [dim * 6, dim * 4, 3, 1, False],
            [dim * 4, dim * 4, 3, 1, True]
        ]
        layer = []
        for i in range(len(self.cfg)):
            layer.append(nn.Conv2d(in_channels=self.cfg[i][0], out_channels=self.cfg[i][1], kernel_size=self.cfg[i][2],
                                   padding=self.cfg[i][3]))
            layer.append(nn.ReLU(inplace=True))
            layer.append(nn.MaxPool2d(kernel_size=3, stride=2)) if self.cfg[i][4] == True else None

        self.First_step = nn.Sequential(*layer)
        self.Second_step = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(186624, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_class),
        )

    def forward(self, x):
        x = self.First_step(x)
        x = self.Second_step(x)
        return x

    def return_transform_info(self):
        return self.transform_info
