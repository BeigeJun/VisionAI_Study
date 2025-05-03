import torch.nn as nn
from torchvision import transforms
from F_Model_Zoo.Models.Classification.Convolution_Modules import conv_n_layer_block
from F_Model_Zoo.Models.Util.ModelBase import modelbase


class VGGNet(modelbase):
    def __init__(self, dim=64, num_class=10):
        super().__init__()

        self.transform_info = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        self.cfg = [
            [3, dim, 2],
            [dim, dim * 2, 2],
            [dim * 2, dim * 4, 3],
            [dim * 4, dim * 8, 3],
            [dim * 8, dim * 8, 3]
        ]
        layer = []
        for i in range(len(self.cfg)):
            layer.append(conv_n_layer_block(self.cfg[i][0], self.cfg[i][1], self.cfg[i][2]))
        self.First_step = nn.Sequential(*layer)
        self.Second_step = nn.Sequential(
            nn.Linear(dim * 8 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_class),
        )

    def forward(self, x):
        x = self.First_step(x)
        x = x.view(x.size(0), -1)
        x = self.Second_step(x)
        return x

    def return_transform_info(self):
        return self.transform_info
