import torch
import torch.nn as nn
from Model_Zoo.Models.Util.ModelBase import modelbase
from torchvision import transforms


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
        self.num_repeats = Repeat

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
            # Input, Output, Kernel, Stride, Residual_repeat
            [3, 32, 3, 1, 0],
            [32, 64, 3, 2, 1],
            [64, 128, 3, 2, 2],
            [128, 256, 3, 2, 8],
            [256, 512, 3, 2, 8],
            [512, 1024, 3, 2, 4]]

        self.layers = []
        for i, (Input, Output, Kernel, Stride, Residual_repeat) in enumerate(self.cfg):
            self.layers.append(BasicCNNBlock(Input=Input, Output=Output, kernel_size=Kernel, stride=Stride, padding=1))
            self.layers.append(ResidualBlock(In_Output=Output, Repeat=Residual_repeat)) if Residual_repeat != 0 else None

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_repeats in [8, 8, 4]:
                outputs.append(x)
        return outputs


class YoloV3(modelbase):
    def __init__(self, num_classes=80):
        super().__init__()
        self.transform_info = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.Backbone = DarkNet53()
        self.num_classes = num_classes
        self.Input = 1024
        self.cfg =[
            #Output, Kernel, ScalePrediction or UpSample
            [512, 1, "X"],
            [1024, 3, "S"],
            [256, 1, "U"],
            [256, 3, "X"],
            [512, 3, "S"],
            [128, 1, "U"],
            [128, 1, "X"],
            [256, 3, "S"]]
        self.layers = self.create_layers()
        print("")

    def create_layers(self):
        layers = []
        Input=self.Input
        for Output, Kernel, layer_type in self.cfg:
            layers.append(BasicCNNBlock(Input, Output, kernel_size=Kernel, stride=1, padding=Kernel // 2))
            Input = Output
            if layer_type == "S":
                layers.append(ResidualBlock(Input, use_Residual=False, Repeat=1))
                layers.append(BasicCNNBlock(Input, Input//2, kernel_size=Kernel, padding=Kernel // 2))
                layers.append(ScalePrediction(Input//2, self.num_classes))
                Input = Input // 2
            elif layer_type == "U":
                layers.append(nn.Upsample(scale_factor=2))
                Input = Input * 3
        return layers

    def forward(self, x):
        outputs = []

        backbone_outputs = self.Backbone(x)
        x = backbone_outputs[-1]
        backbone_outputs.pop()
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, nn.Upsample):
                x = torch.cat([x, backbone_outputs[-1]], dim=1)
                backbone_outputs.pop()

        return outputs

    def return_transform_info(self):
        return self.transform_info
