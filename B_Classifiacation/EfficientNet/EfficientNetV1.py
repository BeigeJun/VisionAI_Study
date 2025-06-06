import math
from B_Classifiacation.Util.Util import *
from B_Classifiacation.Util.Draw_Graph import *


class EfficientNetV1(nn.Module):
    def __init__(self, depth_mult=1.0, width_mult=1.0, resize=1.0, crop=1.0, drop=0.2, num_class=10):
        super().__init__()

        self.transform_info = transforms.Compose([
            transforms.Resize((int(224*resize), int(224*resize))),
            transforms.CenterCrop(int(224*resize*crop)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.configs = {
            # Output, Expansion, Kernel, Stride, Repeat
            'B0': [
                [16, 1, 3, 1, 1],
                [24, 6, 3, 2, 2],
                [40, 6, 5, 2, 2],
                [80, 6, 3, 2, 3],
                [112, 6, 5, 1, 3],
                [192, 6, 5, 2, 4],
                [320, 6, 3, 1, 1]
            ],
            'B1': [
                [16, 1, 3, 1, 1],
                [24, 6, 3, 2, 2],
                [40, 6, 5, 2, 2],
                [80, 6, 3, 2, 3],
                [112, 6, 5, 1, 4],
                [192, 6, 5, 2, 5],
                [320, 6, 3, 1, 2]
            ],
            'B2': [
                [16, 1, 3, 1, 1],
                [24, 6, 3, 2, 2],
                [40, 6, 5, 2, 2],
                [88, 6, 3, 2, 4],
                [120, 6, 5, 1, 4],
                [208, 6, 5, 2, 5],
                [352, 6, 3, 1, 2]
            ],
            'B3': [
                [16, 1, 3, 1, 1],
                [24, 6, 3, 2, 3],
                [48, 6, 5, 2, 3],
                [96, 6, 3, 2, 4],
                [136, 6, 5, 1, 5],
                [232, 6, 5, 2, 6],
                [384, 6, 3, 1, 2]
            ],
            'B4': [
                [16, 1, 3, 1, 1],
                [24, 6, 3, 2, 3],
                [56, 6, 5, 2, 4],
                [112, 6, 3, 2, 5],
                [160, 6, 5, 1, 6],
                [272, 6, 5, 2, 7],
                [448, 6, 3, 1, 3]
            ],
            'B5': [
                [16, 1, 3, 1, 1],
                [24, 6, 3, 2, 3],
                [64, 6, 5, 2, 4],
                [128, 6, 3, 2, 5],
                [176, 6, 5, 1, 7],
                [304, 6, 5, 2, 8],
                [512, 6, 3, 1, 3]
            ],
            'B6': [
                [16, 1, 3, 1, 1],
                [24, 6, 3, 2, 4],
                [72, 6, 5, 2, 5],
                [144, 6, 3, 2, 6],
                [200, 6, 5, 1, 8],
                [344, 6, 5, 2, 9],
                [576, 6, 3, 1, 4]
            ],
            'B7': [
                [16, 1, 3, 1, 1],
                [24, 6, 3, 2, 4],
                [80, 6, 5, 2, 5],
                [160, 6, 3, 2, 6],
                [224, 6, 5, 1, 8],
                [384, 6, 5, 2, 9],
                [640, 6, 3, 1, 4]
            ]
        }

        self.config = self.configs['B0']

        in_put = _make_divisible(32 * width_mult, divisor=8)
        self.First_step = nn.Sequential(
            nn.Conv2d(3, in_put, 3, 2, 1),
            nn.BatchNorm2d(in_put),
            nn.SiLU(inplace=True)
        )
        layers = []
        for Output, Expansion, Kernel, Stride, Repeat in self.config:
            repeat = math.ceil(Repeat*depth_mult)#깊이 조정
            for i in range(repeat):
                stride = Stride if i == 0 else 1 #repeat만큼 반복할 때 처음만 변경사항. 나머진 1로 고정
                output = _make_divisible(Output*width_mult, 8) #너비 조정
                expansion_channel = _make_divisible(in_put * Expansion, 8)
                layers.append(MBconv(in_put, output, Kernel, stride, expansion_channel))
                in_put = output

        self.Second_step = nn.Sequential(*layers)

        last_output = _make_divisible(int(1280 * width_mult), 8)
        self.Third_step = nn.Sequential(
            nn.Conv2d(in_put, last_output, 1),
            nn.BatchNorm2d(last_output),
            nn.SiLU(inplace=True)
        )
        self.Fourth_step = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(last_output, num_class)
        )

    def forward(self, x):
        out = self.First_step(x)
        out = self.Second_step(out)
        out = self.Third_step(out)
        out = self.Fourth_step(out)
        return out

    def return_transform_info(self):
        return self.transform_info


class MBconv(nn.Module):
    def __init__(self, in_put, out_put, kernel_size, stride, expansion):
        super().__init__()
        assert stride in [1, 2], "stride must 1 or 2"

        self.use_short_cut = stride == 1 and in_put == out_put
        self.MBconv1 = True if expansion == 1 else False

        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_put, out_channels=expansion, kernel_size=1, stride=1),
            nn.BatchNorm2d(expansion),
            Swish()
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=expansion, out_channels=expansion, kernel_size=kernel_size,
                      stride=stride, groups=expansion, padding=(kernel_size-1) // 2),
            nn.BatchNorm2d(expansion),
            Swish()
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


class SEBlock(nn.Module):
    def __init__(self, in_put, reduce=4):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_put, max(1, int(in_put//reduce))),
            Swish(),
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


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, '..', 'Util', 'config.yaml')
    yaml_path = os.path.normpath(yaml_path)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetV1(num_class=config['num_class']).to(device)

    graph = Draw_Graph(save_path=config['save_path'], n_patience=config['patience'])

    transform_info = model.return_transform_info()
    train_loader, validation_loader, test_loader = classification_data_loader(config['load_path'],
                                                                              config['batch_size'], transform_info)

    train_model(device=device, model=model, epochs=config['epoch'], patience=config['patience'], train_loader=train_loader,
                val_loader=validation_loader, test_loader=test_loader, lr=0.001, graph=graph)

if __name__ == "__main__":
    main()
