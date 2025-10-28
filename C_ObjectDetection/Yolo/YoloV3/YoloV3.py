import os
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import random_split
from G_Model_Zoo.Models.Util.ModelBase import modelbase
from C_ObjectDetection.Yolo.YoloV3.DataLoader import YoloV3DataLoader, train_transforms , test_transforms
from C_ObjectDetection.Util.Draw_Graph import Draw_Graph
from C_ObjectDetection.Yolo.YoloV3.Trainer import train_model
from C_ObjectDetection.Yolo.YoloV3.Util import test_yolov3_inference


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

        self.layers = nn.ModuleList()
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
        layers = nn.ModuleList()
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


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, '..', '..', 'Util', 'config.yaml')
    yaml_path = os.path.normpath(yaml_path)

    util_path = os.path.dirname(yaml_path)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloV3(num_classes=config['num_class']).to(device)

    graph = Draw_Graph(model=model, save_path=config['save_path'], patience=config['patience'])

    transform_info = model.return_transform_info()


    train_validation_set = YoloV3DataLoader(config['traincsvfile_path'], img_dir=config['IMG_DIR'],
                                            label_dir=config['LABEL_DIR'], image_size=416, C=20,
                                            transform=train_transforms)

    test_set = YoloV3DataLoader(config['testcsvfile_path'], img_dir=config['IMG_DIR'],
                                label_dir=config['LABEL_DIR'], image_size=416, C=20,
                                transform=test_transforms)

    train_set_num = int(0.8 * len(train_validation_set))
    validation_set_num = len(train_validation_set) - train_set_num

    train_set, validation_set = random_split(train_validation_set, [train_set_num, validation_set_num])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

    #train_model(device=device, model=model, train_loader=train_loader, val_loader=validation_loader, test_loader=test_loader,
    #             graph=graph, epochs=config['epoch'], lr=0.001, patience=config['patience'], graph_update_epoch = 2)
    model.load_state_dict(torch.load('D:/0. Model_Save_Folder/Best_Accuracy_Train.pth', map_location=device))

    test_yolov3_inference(model=model, loader=test_loader, device=device)


if __name__ == "__main__":
    main()