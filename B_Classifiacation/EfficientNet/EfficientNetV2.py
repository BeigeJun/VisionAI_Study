import math
from B_Classifiacation.Util.Util import *
from B_Classifiacation.Util.Draw_Graph import *


class EfficientNetV2(nn.Module):
    def __init__(self, model_name='Small', width_mult=1.0, depth_mult=1.0,
                 resize=1.0, crop=1.0, drop=0.2, num_class=10):
        super().__init__()

        self.transform_info = transforms.Compose([
            transforms.Resize((int(224 * resize), int(224 * resize))),
            transforms.CenterCrop(int(224 * resize * crop)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.configs = {
            'Small': [
                # Block_type, Output, Expansion, Kernel, Stride, Repeat, SE_ratio
                ['fused', 24, 1, 3, 1, 2, None],
                ['fused', 48, 4, 3, 2, 4, None],
                ['fused', 64, 4, 3, 2, 4, None],
                ['mbconv', 128, 4, 3, 2, 6, 0.25],
                ['mbconv', 160, 6, 3, 1, 9, 0.25],
                ['mbconv', 256, 6, 3, 2, 15, 0.25]
            ],
            'Medium' : [
                ['fused', 24, 1, 3, 1, 3, None],
                ['fused', 48, 4, 3, 2, 5, None],
                ['fused', 80, 4, 3, 2, 5, None],
                ['mbconv', 160, 4, 3, 2, 9, 0.25],
                ['mbconv', 176, 6, 3, 1, 14, 0.25],
                ['mbconv', 304, 6, 3, 2, 18, 0.25]
            ],
            'Large' : [
                ['fused', 32, 1, 3, 1, 4, None],
                ['fused', 64, 4, 3, 2, 7, None],
                ['fused', 96, 4, 3, 2, 7, None],
                ['mbconv', 192, 4, 3, 2, 16, 0.25],
                ['mbconv', 224, 6, 3, 1, 23, 0.25],
                ['mbconv', 384, 6, 3, 2, 27, 0.25]
            ]
        }

        self.config = self.configs[model_name]

        in_put = _make_divisible(24 * width_mult, 8)
        self.First_step = nn.Sequential(
            nn.Conv2d(3, in_put, 3, 2, 1),
            nn.BatchNorm2d(in_put),
            Swish()
        )

        layers = []
        for block_type, Output, Expansion, Kernel, Stride, Repeat, se_ratio in self.config:
            repeat = math.ceil(Repeat * depth_mult)
            for i in range(repeat):
                stride = Stride if i == 0 else 1
                output = _make_divisible(Output * width_mult, 8)

                if block_type == 'fused':
                    layers.append(FusedMBConv(in_put, output, Kernel, stride, Expansion, se_ratio))
                else:
                    layers.append(MBconv(in_put, output, Kernel, stride, Expansion, se_ratio))

                in_put = output

        self.Second_step = nn.Sequential(*layers)

        last_output = _make_divisible(1280 * width_mult, 8)
        self.Third_step = nn.Sequential(
            nn.Conv2d(in_put, last_output, 1),
            nn.BatchNorm2d(last_output),
            Swish()
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
        return self.Fourth_step(out)

    def return_transform_info(self):
        return self.transform_info


class FusedMBConv(nn.Module):
    def __init__(self, in_put, out_put, kernel_size, stride, expansion, se_ratio=None):
        super().__init__()
        self.use_shortcut = (stride == 1) and (in_put == out_put)
        hidden_dim = in_put * expansion

        layers = []
        layers.append(nn.Conv2d(in_put, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=1))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(Swish())

        if se_ratio:
            layers.append(SEBlock(hidden_dim, se_ratio))

        layers.append(nn.Conv2d(hidden_dim, out_put, 1))
        layers.append(nn.BatchNorm2d(out_put))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.block(x)
        return self.block(x)


class MBconv(nn.Module):
    def __init__(self, in_put, out_put, kernel_size, stride, expansion, se_ratio=0.25):
        super().__init__()
        self.use_shortcut = (stride == 1) and (in_put == out_put)
        hidden_dim = in_put * expansion

        layers = []
        if expansion != 1:
            layers.append(nn.Conv2d(in_put, hidden_dim, 1))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(Swish())

        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(Swish())
        layers.append(SEBlock(hidden_dim, se_ratio))

        layers.append(nn.Conv2d(hidden_dim, out_put, 1))
        layers.append(nn.BatchNorm2d(out_put))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.block(x)
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        reduced_channels = max(1, int(in_channels * se_ratio))
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            Swish(),
            nn.Linear(reduced_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        se = self.squeeze(x).view(bs, c)
        se = self.excitation(se).view(bs, c, 1, 1)
        return x * se


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, '..', 'Util', 'config.yaml')
    yaml_path = os.path.normpath(yaml_path)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetV2(num_class=config['num_class']).to(device)

    graph = Draw_Graph(save_path=config['save_path'], n_patience=config['patience'])

    transform_info = model.return_transform_info()
    train_loader, validation_loader, test_loader = classification_data_loader(config['load_path'],
                                                                              config['batch_size'], transform_info)

    train_model(device=device, model=model, epochs=config['epoch'], patience=config['patience'], train_loader=train_loader,
                val_loader=validation_loader, test_loader=test_loader, lr=0.001, graph=graph)

if __name__ == "__main__":
    main()
