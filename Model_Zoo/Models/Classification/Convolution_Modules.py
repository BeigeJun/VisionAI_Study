import torch.nn as nn


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
    layers.append(nn.Conv2d(in_put*expansion, in_put*expansion, kernel_size=kernel, stride=stride, padding=1,
                            groups=in_put*expansion, bias=False))
    layers.append(nn.BatchNorm2d(in_put*expansion))
    if use_relu_hswish == True:
        layers.append(nn.ReLU6())
    elif use_relu_hswish == False:
        layers.append(h_swish())
    return nn.Sequential(*layers)


def conv_separable(in_put, out_put, in_put_expansion=1, out_put_expansion=1, use_relu6=True, use_relu_hswish=True):
    layers = []
    layers.append(nn.Conv2d(in_put*in_put_expansion, out_put*out_put_expansion, kernel_size=1, stride=1, padding=0,
                            bias=False))
    layers.append(nn.BatchNorm2d(out_put*out_put_expansion))
    if use_relu6:
        if use_relu_hswish == True:
            layers.append(nn.ReLU6())
        elif use_relu_hswish == False:
            layers.append(h_swish())
    return nn.Sequential(*layers)


def se_module(in_put, out_put, expansion):
    layers = []
    layers.append(nn.AdaptiveAvgPool2d(1)),
    layers.append(nn.Conv2d(in_put*expansion, _make_divisible(in_put*expansion//4), 1, 1))
    layers.append(nn.ReLU6()),
    layers.append(nn.Conv2d(_make_divisible(in_put*expansion//4), out_put, 1, 1))
    layers.append(h_swish())
    return nn.Sequential(*layers)


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_put, out_put, stride=1):
        super().__init__()
        self.Depth_Wise = conv_depth_wise(in_put=in_put, stride=stride)
        self.Point_Wise = conv_separable(in_put=in_put, out_put=out_put)

    def forward(self, x):
        x = self.Depth_Wise(x)
        x = self.Point_Wise(x)
        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_put, out_put, expansion, kernel, stride, se=False, re=True):
        super().__init__()
        assert stride in [1, 2], "stride must 1 or 2"
        self.use_short_cut = stride == 1 and in_put == out_put
        self.Point_Wise1 = conv_separable(in_put=in_put, out_put=in_put, out_put_expansion=expansion, use_relu_hswish=re)
        self.Depth_Wise = conv_depth_wise(in_put=in_put, expansion=expansion, kernel=kernel, stride=stride)
        self.SE_modul = se_module(in_put=in_put, out_put=in_put, expansion=expansion)
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


























