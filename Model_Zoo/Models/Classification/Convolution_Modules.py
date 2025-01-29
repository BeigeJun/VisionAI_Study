import torch.nn as nn
import torch

#------------------------------------------------MobileNet--------------------------------------------------------------


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


class h_swish(nn.Module):
    def __init__(self):
        super(h_swish, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return x * (self.relu6(x + 3) / 6)


def conv_depth_wise(in_put, expansion=1, kernel=3, stride=1, use_relu_hswish=True):
    layers = []
    padding = kernel // 2
    layers.append(nn.Conv2d(int(in_put*expansion), int(in_put*expansion), kernel_size=kernel, stride=stride,
                            padding=padding, groups=int(in_put*expansion), bias=False))
    layers.append(nn.BatchNorm2d(int(in_put*expansion)))
    if use_relu_hswish == True:
        layers.append(nn.ReLU6())
    elif use_relu_hswish == False:
        layers.append(h_swish())
    return nn.Sequential(*layers)


def conv_separable(in_put, out_put, in_put_expansion=1, out_put_expansion=1, use_relu6=True, use_relu_hswish=True):
    layers = []
    layers.append(nn.Conv2d(int(in_put*in_put_expansion), int(out_put*out_put_expansion), kernel_size=1, stride=1, padding=0,
                            bias=False))
    layers.append(nn.BatchNorm2d(int(out_put*out_put_expansion)))
    if use_relu6:
        if use_relu_hswish == True:
            layers.append(nn.ReLU6())
        elif use_relu_hswish == False:
            layers.append(h_swish())
    return nn.Sequential(*layers)


class SEModule(nn.Module):
    def __init__(self, in_put, reduce=4):
        super(SEModule, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_put, int(in_put//reduce)),
            nn.ReLU(),
            nn.Linear(int(in_put//reduce), in_put),
            h_swish()
        )

    def forward(self, x):
        out = self.squeeze(x)
        out = out.view(out.size(0), -1)
        out = self.excitation(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        return out * x


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_put, out_put, stride=1):
        super().__init__()
        self.Depth_Wise = conv_depth_wise(in_put=in_put, stride=stride)
        self.Point_Wise = conv_separable(in_put=in_put, out_put=out_put)

    def forward(self, x):
        out = self.Depth_Wise(x)
        out = self.Point_Wise(out)
        return out


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_put, out_put, expansion, kernel, stride, se=False, re=True):
        super().__init__()
        assert stride in [1, 2], "stride must 1 or 2"
        self.use_short_cut = stride == 1 and in_put == out_put
        self.Point_Wise1 = conv_separable(in_put=in_put, out_put=in_put, out_put_expansion=expansion, use_relu_hswish=re)
        self.Depth_Wise = conv_depth_wise(in_put=in_put, expansion=expansion, kernel=kernel, stride=stride)
        self.SE_modul = SEModule(in_put=int(in_put*expansion))
        self.Point_Wise2 = conv_separable(in_put=in_put, out_put=out_put, in_put_expansion=expansion, use_relu6=False,
                                          use_relu_hswish=re)
        self.se = se

    def forward(self, x):
        out = self.Point_Wise1(x)
        out = self.Depth_Wise(out)
        if self.se:
            out = self.SE_modul(out)
        out = self.Point_Wise2(out)
        if self.use_short_cut:
            out = out + x
        return out


#------------------------------------------------Resnet-----------------------------------------------------------------


class DownSample(nn.Module):
    def __init__(self, in_put, out_put, stride, expansion=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_put, out_channels=out_put*expansion, kernel_size=1, stride=stride,
                              bias=False)
        self.batch = nn.BatchNorm2d(out_put*expansion)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_put, out_put, stride, expansion):
        super().__init__()
        self.expansion = expansion

        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_put, out_channels=out_put, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_put),
            nn.ReLU()
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=out_put, out_channels=out_put, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_put),
            nn.ReLU()
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_put, out_channels=out_put*expansion, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_put*expansion)
        )

        self.downsample = DownSample(in_put=in_put, out_put=out_put, stride=stride, expansion=expansion)
        self.use_downsample = False
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_put != out_put * self.expansion:
            self.use_downsample = True

    def forward(self, x):
        out = self.conv1x1_1(x)
        out = self.conv3x3(out)
        out = self.conv1x1_2(out)

        if self.use_downsample:
            down_out = self.downsample(x)
            out += down_out
        out = self.relu(out)
        return out


#---------------------------------------------------EfficientNet--------------------------------------------------------

class SEBlock(nn.Module):
    def __init__(self, in_put, reduce=4):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_put, max(1, int(in_put//reduce))),
            nn.SiLU(),
            nn.Linear(max(1, int(in_put//reduce)), in_put),
            nn.Sigmoid()
        )

    def forward(self, x):
        se = self.squeeze(x)
        se = torch.flatten(se, 1)
        se = self.excitation(se)
        se = se.unsqueeze(dim=2).unsqueeze(dim=3)
        out = se * x
        return out


class MBconv(nn.Module):
    def __init__(self, in_put, out_put, kernel_size, stride, expansion):
        super().__init__()
        assert stride in [1, 2], "stride must 1 or 2"

        self.use_short_cut = stride == 1 and in_put == out_put
        self.MBconv1 = True if expansion == 1 else False

        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_put, out_channels=expansion, kernel_size=1, stride=1),
            nn.BatchNorm2d(expansion),
            nn.ReLU()
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=expansion, out_channels=expansion, kernel_size=kernel_size,
                      stride=stride, groups=expansion, padding=(kernel_size-1) // 2),
            nn.BatchNorm2d(expansion),
            nn.ReLU()
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels=expansion, out_channels=out_put, kernel_size=1),
            nn.BatchNorm2d(out_put)
        )
        self.SE = SEBlock(expansion)

    def forward(self, x):
        identity = x
        if not self.MBconv1:
            x = self.conv1x1_1(x)
        x = self.conv3x3(x)
        x = self.SE(x)
        x = self.conv1x1_2(x)
        if self.use_short_cut:
            return x + identity
        return x

