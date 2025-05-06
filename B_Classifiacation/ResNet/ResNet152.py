from B_Classifiacation.Util.Util import *
from B_Classifiacation.Util.Draw_Graph import *

class ResNet152(nn.Module):
    def __init__(self,  num_class=10):
        super().__init__()

        self.transform_info = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.expansion = 4

        self.configs = [
            #Out, Stride, Repeat
            [64, 1, 3],
            [128, 2, 8],
            [256, 2, 36],
            [512, 2, 3]
        ]

        self.First_step = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU6()
        )

        self.Max_Pool = nn.MaxPool2d(3, 2, 1)

        layer = []
        prev_channels = 64

        for i in range(len(self.configs)):
            in_channels = prev_channels
            out_channels = self.configs[i][0]

            for j in range(self.configs[i][2]):
                layer.append(ResidualBlock(in_channels, out_channels,
                                           stride=self.configs[i][1] if j == 0 else 1,
                                           expansion=self.expansion))
                in_channels = out_channels * self.expansion
            prev_channels = out_channels * self.expansion

        self.Second_step = nn.Sequential(*layer)

        self.Third_step = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_class)
        )

    def forward(self, x):
        x = self.First_step(x)
        x = self.Max_Pool(x)
        x = self.Second_step(x)
        x = self.Third_step(x)
        return x

    def return_transform_info(self):
        return self.transform_info


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

        if stride != 1 or in_put != out_put * expansion:
            self.use_downsample = True

    def forward(self, x):
        in_put = x
        out = self.conv1x1_1(in_put)
        out = self.conv3x3(out)
        out = self.conv1x1_2(out)

        if self.use_downsample:
            in_put = self.downsample(in_put)
        out += in_put
        out = self.relu(out)
        return out

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, '..', 'Util', 'config.yaml')
    yaml_path = os.path.normpath(yaml_path)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet152(num_class=config['num_class']).to(device)

    graph = Draw_Graph(save_path=config['save_path'], n_patience=config['patience'])

    transform_info = model.return_transform_info()
    train_loader, validation_loader, test_loader = classification_data_loader(config['load_path'],
                                                                              config['batch_size'], transform_info)

    train_model(device=device, model=model, epochs=config['epoch'], patience=config['patience'], train_loader=train_loader,
                val_loader=validation_loader, test_loader=test_loader, lr=0.001, graph=graph)

if __name__ == "__main__":
    main()
