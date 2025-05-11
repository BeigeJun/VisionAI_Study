from B_Classifiacation.Util.Util import *
from B_Classifiacation.Util.Draw_Graph import *

class MobileNetV1(nn.Module):
    def __init__(self, alpha=1.0, num_class=10):
        super().__init__()

        self.transform_info = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.configs = [
            [2, 32, 32, 1], [1, 32, 64, 1],
            [2, 64, 64, 2], [1, 64, 128, 1],
            [2, 128, 128, 1], [1, 128, 128, 1],
            [2, 128, 128, 1], [1, 128, 128, 1],
            [2, 128, 128, 2], [1, 128, 256, 1],
            [2, 256, 256, 1], [1, 256, 256, 1],
            [2, 256, 256, 2], [1, 256, 512, 1],

            [2, 512, 512, 1], [1, 512, 512, 1],
            [2, 512, 512, 1], [1, 512, 512, 1],
            [2, 512, 512, 1], [1, 512, 512, 1],
            [2, 512, 512, 1], [1, 512, 512, 1],
            [2, 512, 512, 1], [1, 512, 512, 1],
            [2, 512, 512, 2], [1, 512, 1024, 1],
            [2, 1024, 1024, 1], [1, 1024, 1024, 1]
        ]

        self.First_Step = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        layers = []
        for Type, Input, OutPut, Stride in self.configs:
            Input = int(Input * alpha)
            OutPut = int(OutPut * alpha)

            if Type == 2:
                layers.append(DepthWiseSeparableConv(Input, OutPut, Stride))
            elif Type == 1:
                layers.append(conv_separable(Input, OutPut))

        self.Second_Step = nn.Sequential(*layers)

        self.Third_Step = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(int(1024 * alpha), num_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.First_Step(x)
        x = self.Second_Step(x)
        x = self.Third_Step(x)
        return x

    def return_transform_info(self):
        return self.transform_info


def conv_separable(in_put, out_put):
    layers = []
    layers.append(nn.Conv2d(in_put, out_put, kernel_size=1, stride=1, padding=0, bias=False))
    layers.append(nn.BatchNorm2d(out_put))
    layers.append(nn.ReLU6())

    return nn.Sequential(*layers)


def conv_depth_wise(in_put, kernel=3, stride=1):
    layers = []
    padding = kernel // 2
    layers.append(nn.Conv2d(in_put, in_put, kernel_size=kernel, stride=stride, padding=padding, groups=in_put, bias=False))
    layers.append(nn.BatchNorm2d(in_put))
    layers.append(nn.ReLU6())

    return nn.Sequential(*layers)


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_put, out_put, stride=1):
        super().__init__()
        self.Depth_Wise = conv_depth_wise(in_put=in_put, stride=stride)
        self.Point_Wise = conv_separable(in_put=in_put, out_put=out_put)

    def forward(self, x):
        out = self.Depth_Wise(x)
        out = self.Point_Wise(out)
        return out

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, '..', 'Util', 'config.yaml')
    yaml_path = os.path.normpath(yaml_path)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV1(num_class=config['num_class']).to(device)

    graph = Draw_Graph(save_path=config['save_path'], n_patience=config['patience'])

    transform_info = model.return_transform_info()
    train_loader, validation_loader, test_loader = classification_data_loader(config['load_path'],
                                                                              config['batch_size'], transform_info)

    train_model(device=device, model=model, epochs=config['epoch'], patience=config['patience'], train_loader=train_loader,
                val_loader=validation_loader, test_loader=test_loader, lr=0.001, graph=graph)

if __name__ == "__main__":
    main()
