import os
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import random_split
from G_Model_Zoo.Models.Util.ModelBase import modelbase
from C_ObjectDetection.Yolo.YoloV1.DataLoader import YoloV1DataLoader, Compose
from C_ObjectDetection.Util.Draw_Graph import Draw_Graph
from C_ObjectDetection.Yolo.YoloV1.Util import test_yolov1_inference
from C_ObjectDetection.Yolo.YoloV1.Trainer import train_model

class Yolov1(modelbase):
    def __init__(self, split_size, num_boxes, num_classes=20):
        super().__init__()

        self.transform_info = [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ]

        self.cfg = [
            # Input, Output, Kernel, Stride, Pool
            [3, 64, 7, 2, True],
            [64, 192, 3, 1, True],
            [192, 128, 1, 1, False],
            [128, 256, 3, 1, False],
            [256, 256, 1, 1, False],
            [256, 512, 3, 1, True],
            [512, 256, 1, 1, False],
            [256, 512, 3, 1, False],
            [512, 256, 1, 1, False],
            [256, 512, 3, 1, False],
            [512, 256, 1, 1, False],
            [256, 512, 3, 1, False],
            [512, 256, 1, 1, False],
            [256, 512, 3, 1, False],
            [512, 512, 1, 1, False],
            [512, 1024, 3, 1, True],
            [1024, 512, 1, 1, False],
            [512, 1024, 3, 1, False],
            [1024, 512, 1, 1, False],
            [512, 1024, 3, 1, False],
            [1024, 1024, 3, 1, False],
            [1024, 1024, 3, 2, False],
            [1024, 1024, 3, 1, False],
            [1024, 1024, 3, 1, False]
        ]
        layers = []
        for Input, Output, Kernel, Stride, Pool in self.cfg:
            layers.append(nn.Conv2d(in_channels=Input, out_channels=Output,
                                    kernel_size=Kernel, stride=Stride, padding=Kernel//2))
            layers.append(nn.BatchNorm2d(Output))
            layers.append(nn.LeakyReLU(0.1))
            if Pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.First_layer = nn.Sequential(*layers)
        self.Second_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * split_size * split_size, 496),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            #마지막 7*7*30의 아웃풋이 7*7로 나눠진 영역의 바운딩박스 2개 * (x, y, w, h, confidence) + 각 클래스의 확률
            nn.Linear(496, split_size * split_size * (5 * num_boxes + num_classes))
        )

    def forward(self, x):
        out = self.First_layer(x)
        return self.Second_layer(out)

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
    model = Yolov1(split_size=7, num_boxes=2, num_classes=config['num_class']).to(device)

    graph = Draw_Graph(model=model, save_path=config['save_path'], patience=config['patience'])

    transform_info = model.return_transform_info()

    transform = Compose(transform_info)
    train_validation_set = YoloV1DataLoader(config['traincsvfile_path'], transform=transform,
                                            img_dir=config['IMG_DIR'], label_dir=config['LABEL_DIR'])
    test_set = YoloV1DataLoader(config['testcsvfile_path'], transform=transform,
                                img_dir=config['IMG_DIR'], label_dir=config['LABEL_DIR'])

    train_set_num = int(0.8 * len(train_validation_set))
    validation_set_num = len(train_validation_set) - train_set_num

    train_set, validation_set = random_split(train_validation_set, [train_set_num, validation_set_num])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

    # train_model(device=device, model=model, train_loader=train_loader, val_loader=validation_loader, test_loader=test_loader,
    #             graph=graph, epochs=config['epoch'], lr=0.001, patience=config['patience'], graph_update_epoch = 2)
    model.load_state_dict(torch.load('D:/0. Model_Save_Folder/Bottom_Loss_Validation.pth', map_location=device))
    test_yolov1_inference(model=model, loader=test_loader, device=device)

if __name__ == "__main__":
    main()
