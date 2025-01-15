import torch.nn as nn


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_put, out_put, stride=1):
        super.__init__()
        self.Depth_Wise = nn.Sequential(
            nn.Conv2d(in_put, in_put, 3, stride=stride, padding=1, groups=in_put, bias=False),
            nn.BatchNorm2d(in_put),
            nn.ReLU6()
        )
        self.Point_Wise = nn.Sequential(
            nn.Conv2d(in_put, out_put, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_put),
            nn.ReLU6()
        )

    def forward(self, x):
        x = self.Depth_Wise(x)
        x = self.Point_Wise(x)
        return x
