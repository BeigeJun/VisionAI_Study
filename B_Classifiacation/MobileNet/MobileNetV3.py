from B_Classifiacation.Util.Util import *
from B_Classifiacation.Util.Draw_Graph import *

class MobileNetV3(nn.Module):
    def __init__(self, model_type='large', alpha=1.0, num_class=10):
        super().__init__()

        self.transform_info = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        large = [
            # Input, Output, Expansion, Kernel, Stride, SE, UseHS
            [16, 16, 1, 3, 1, False, False],
            [16, 24, 4, 3, 2, False, False],
            [24, 24, 3, 3, 1, False, False],
            [24, 40, 3, 5, 2, False, True],
            [40, 40, 3, 5, 1, False, True],
            [40, 40, 3, 5, 1, False, True],
            [40, 80, 6, 3, 2, True, False],
            [80, 80, 2.5, 3, 1, True, False],
            [80, 80, 2.4, 3, 1, True, False],
            [80, 80, 2.4, 3, 1, True, False],
            [80, 112, 6, 3, 1, True, True],
            [112, 112, 6, 3, 1, True, True],
            [112, 160, 6, 5, 2, True, True],
            [160, 160, 6, 5, 1, True, True],
            [160, 160, 6, 5, 1, True, True]
        ]
        small = [
            [16, 16, 1, 3, 2, False, True],
            [16, 24, 4, 3, 2, False, False],
            [24, 24, 11.0 / 3.0, 3, 1, False, False],
            [24, 40, 4, 5, 2, True, True],
            [40, 40, 6, 5, 1, True, True],
            [40, 40, 6, 5, 1, True, True],
            [40, 48, 3, 5, 1, True, True],
            [48, 48, 3, 5, 1, True, True],
            [48, 96, 6, 5, 2, True, True],
            [96, 96, 6, 5, 1, True, True],
            [96, 96, 6, 5, 1, True, True],
        ]

        if model_type == 'small':
            model = small
            last_linear = 1024
        elif model_type == 'large':
            model = large
            last_linear = 1280
        else:
            model = None
            last_linear = 0

        in_put = _make_divisible(16 * alpha)
        out_put = 0
        self.First_Step = nn.Sequential(
            nn.Conv2d(3, in_put, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(16*alpha)),
            nn.ReLU6(inplace=True)
        )
        layers = []
        for i in range(len(model)):
            out_put = _make_divisible(model[i][1] * alpha)
            layers.append(InvertedResidual(
                in_put,
                out_put,
                expansion=model[i][2],
                kernel=model[i][3],
                stride=model[i][4],
                se=model[i][5],
                use_hs=model[i][6]
            ))
            in_put = out_put

        self.Second_Step = nn.Sequential(*layers)

        self.Third_Step = nn.Sequential(
            nn.Conv2d(in_put, out_put * 6, 1, 1),
            nn.BatchNorm2d(out_put * 6),
            h_swish(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_put * 6, last_linear),
            h_swish(),
            nn.Linear(last_linear, num_class)
            )

    def forward(self, x):
        x = self.First_Step(x)
        x = self.Second_Step(x)
        x = self.Third_Step(x)
        return x

    def return_transform_info(self):
        return self.transform_info



def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self, in_put, out_put, expansion, kernel, stride, se, use_hs):
        super().__init__()
        assert stride in [1, 2]

        self.stride = stride
        hidden_dim = int(round(in_put * expansion))
        self.use_res_connect = self.stride == 1 and in_put == out_put

        layers = []
        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_put, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU6(inplace=True)
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel, stride, kernel // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if use_hs else nn.ReLU6(inplace=True)
        ])

        if se:
            layers.append(SEModule(hidden_dim))

        layers.extend([
            nn.Conv2d(hidden_dim, out_put, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_put)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class h_swish(nn.Module):
    def __init__(self):
        super(h_swish, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return x * (self.relu6(x + 3) / 6)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu6(x + 3) / 6


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, '..', 'Util', 'config.yaml')
    yaml_path = os.path.normpath(yaml_path)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3(num_class=config['num_class']).to(device)

    graph = Draw_Graph(save_path=config['save_path'], n_patience=config['patience'])

    transform_info = model.return_transform_info()
    train_loader, validation_loader, test_loader = classification_data_loader(config['load_path'],
                                                                              config['batch_size'], transform_info)

    train_model(device=device, model=model, epochs=config['epoch'], patience=config['patience'], train_loader=train_loader,
                val_loader=validation_loader, test_loader=test_loader, lr=0.001, graph=graph)

if __name__ == "__main__":
    main()
