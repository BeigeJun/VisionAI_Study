import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
import xmltodict
from PIL import Image
import numpy as np
from tqdm import tqdm

class YOLO(torch.nn.Module):
    def __init__(self, VGG16):
        super(YOLO, self).__init__()
        self.backbone = VGG16
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1470)
        )

        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)

        for m in self.linear.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)


    def forward(self, x):
        out = self.backbone(x)
        out = self.conv(out)
        out = self.linear(out)
        out = torch.reshape(out, (-1, 7, 7, 30))
        return out
