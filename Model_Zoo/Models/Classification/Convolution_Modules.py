import torch.nn as nn


def conv_depth_wise(in_put, expansion=1, stride=1):
    layers = []
    layers.append(nn.Conv2d(in_put*expansion, in_put*expansion, 3, stride=stride, padding=1, groups=in_put*expansion,
                            bias=False))
    layers.append(nn.BatchNorm2d(in_put*expansion))
    layers.append(nn.ReLU6())
    return nn.Sequential(*layers)


def conv_separable(in_put, out_put, in_put_expansion=1, out_put_expansion=1, use_relu6=True):
    layers = []
    layers.append(nn.Conv2d(in_put*in_put_expansion, out_put*out_put_expansion, 1, stride=1, padding=0, bias=False))
    layers.append(nn.BatchNorm2d(out_put*out_put_expansion))
    if use_relu6:
        layers.append(nn.ReLU6())
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
    def __init__(self, in_put, out_put, expansion, stride):
        super().__init__()
        assert stride in [1, 2], "stride must 1 or 2"
        self.use_short_cut = stride == 1 and in_put == out_put
        self.Point_Wise1 = conv_separable(in_put=in_put, out_put=in_put, out_put_expansion=expansion)
        self.Depth_Wise = conv_depth_wise(in_put=in_put, expansion=expansion, stride=stride)
        self.Point_Wise2 = conv_separable(in_put=in_put, out_put=out_put, in_put_expansion=expansion, use_relu6=False)

    def forward(self, x):
        out = self.Point_Wise1(x)
        out = self.Depth_Wise(out)
        out = self.Point_Wise2(out)
        if self.use_short_cut:
            out = out + x
        return out
