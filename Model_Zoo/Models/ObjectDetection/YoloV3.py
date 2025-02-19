import torch
import torch.nn as nn
from Model_Zoo.Models.Util.ModelBase import modelbase


class BasicCNNBlock(nn.Module):
    def __init__(self, Input, Output, Batchnomalize=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=Input, out_channels=Output, bias=not Batchnomalize, **kwargs)
        self.bn = nn.BatchNorm2d(Output)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn = Batchnomalize

    def forward(self, x):
        if self.use_bn:
            out = self.conv(x)
            out = self.bn(out)
            return self.leaky(out)
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, In_Output, use_Residual=True, Repeat=1):
        super().__init__()
        layers = []
        for _ in range(Repeat):
            layers.append(BasicCNNBlock(In_Output, In_Output//2, kernel_size=1))
            layers.append(BasicCNNBlock(In_Output//2, In_Output, kernel_size=3, padding=1))
        self.Residual_step = nn.Sequential(*layers)
        self.use_Residual = use_Residual

    def forward(self, x):
        if self.use_Residual:
            out = x + self.Residual_step(x)
        else:
            out = self.Residual_step(x)
        return out


class ScalePrediction(nn.Module):
    def __init__(self, Input, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            BasicCNNBlock(Input, 2 * Input, kernel_size=3, padding=1),
            BasicCNNBlock(2 * Input, (num_classes + 5) * 3, Batchnomalize=False, kernel_size=1)
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])#Batch, 앵커 박스 수, 클래스 + 5, 높이, 너비
            .permute(0, 1, 3, 4, 2)#순서 바꾸기
        )


class DarkNet53(nn.Module):
    def __init__(self):
        super().__init__()

        self.cfg = [
            # Input, Output, Kernel, Stride
            [3, 32, 3, 1],
            [32, 64, 3, 2],
            [64, 128, 3, 2],
            [128, 256, 3, 2],
            [256, 512, 3, 2],
            [512, 1024, 3, 2]]
        self.residual_repeat = [1, 2, 8, 8, 4]