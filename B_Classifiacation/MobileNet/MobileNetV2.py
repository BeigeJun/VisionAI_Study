from B_Classifiacation.Util.Util import *
from B_Classifiacation.Util.Draw_Graph import *

class MobileNetV2(nn.Module):
    def __init__(self, alpha=1.0, num_class=10):
        super().__init__()

        self.transform_info = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        configs = [
            # Expansion, Output, Repeat, Stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        input_channel = _make_divisible(32 * alpha)

        self.First_Step = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )

        layers = []
        for Expansion, Output, Repeat, Stride in configs:
            output_channel = _make_divisible(Output * alpha)
            for i in range(Repeat):
                stride = Stride if i == 0 else 1
                layers.append(InvertedResidual(in_put=input_channel, out_put=output_channel, stride=stride, expansion=Expansion))
                input_channel = output_channel

        self.Second_Step = nn.Sequential(*layers)

        self.Third_Step = nn.Sequential(
            nn.Conv2d(input_channel, _make_divisible(1280 * alpha), 1, 1, 0, bias=False),
            nn.BatchNorm2d(_make_divisible(1280 * alpha)),
            nn.ReLU6(inplace=True))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(_make_divisible(1280 * alpha), num_class)
        )

    def forward(self, x):
        x = self.First_Step(x)
        x = self.Second_Step(x)
        x = self.Third_Step(x)
        x = self.classifier(x)
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
    def __init__(self, in_put, out_put, stride, expansion):
        super().__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(in_put * expansion))
        self.skip_connection = stride == 1 and in_put == out_put

        layers = []
        if expansion != 1:
            layers.append(nn.Conv2d(in_put, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))


        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))


        layers.append(nn.Conv2d(hidden_dim, out_put, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_put))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.skip_connection:
            return x + self.conv(x)
        return self.conv(x)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, '..', 'Util', 'config.yaml')
    yaml_path = os.path.normpath(yaml_path)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2(num_class=config['num_class']).to(device)

    graph = Draw_Graph(save_path=config['save_path'], n_patience=config['patience'])

    transform_info = model.return_transform_info()
    train_loader, validation_loader, test_loader = classification_data_loader(config['load_path'],
                                                                              config['batch_size'], transform_info)

    train_model(device=device, model=model, epochs=config['epoch'], patience=config['patience'], train_loader=train_loader,
                val_loader=validation_loader, test_loader=test_loader, lr=0.001, graph=graph)

if __name__ == "__main__":
    main()
