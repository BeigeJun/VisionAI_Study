from D_AnomalyDetection.Util.Util import *
from D_AnomalyDetection.Util.Draw_Graph import *

class AutoEncoder(nn.Module):
    def __init__(self, alpha=1.0, num_class=10):
        super().__init__()
        self.resize_size = (256, 256)
        self.transform_info = transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.ToTensor()
        ])

        self.model_name = "AutoEncoder"
        self.network = []
        self.EncoderDecoder = self.create_network(self.resize_size)
        self.Threshold = 0.0

    def create_network(self, resize_size):
        input_channel = 3
        channel = 4
        patch_size = resize_size[0]

        # Encoder part
        self.network.append(nn.Conv2d(input_channel, channel, kernel_size=4, stride=2, padding=1))  # 256->128
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1))  # 128->64
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        if patch_size == 256:
            self.network.append(nn.Conv2d(channel, channel, kernel_size=4, stride=2, padding=1))  # 64->32
            self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1))
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(nn.Conv2d(channel, channel * 2, kernel_size=4, stride=2, padding=1))  # Downsample
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(nn.Conv2d(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1))
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(nn.Conv2d(channel * 2, channel * 4, kernel_size=4, stride=2, padding=1))
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(nn.Conv2d(channel * 4, channel * 2, kernel_size=3, stride=1, padding=1))
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1))
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(nn.Conv2d(channel, 100, kernel_size=8, stride=1, padding=0))  # encoded layer
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        # Decoder part
        self.network.append(nn.ConvTranspose2d(100, channel, kernel_size=8, stride=1, padding=0))
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(nn.Conv2d(channel, channel * 2, kernel_size=3, stride=1, padding=1))
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(nn.Conv2d(channel * 2, channel * 4, kernel_size=3, stride=1, padding=1))
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(nn.ConvTranspose2d(channel * 4, channel * 2, kernel_size=4, stride=2, padding=1))
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(nn.Conv2d(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1))
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(nn.ConvTranspose2d(channel * 2, channel, kernel_size=4, stride=2, padding=1))
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1))
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1))
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        if patch_size == 256:
            self.network.append(nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1))
            self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(nn.ConvTranspose2d(channel, input_channel, kernel_size=4, stride=2, padding=1))
        self.network.append(nn.Sigmoid())

        return nn.Sequential(*self.network)

    def forward(self, x):
        x = self.EncoderDecoder(x)
        return x

    def return_transform_info(self):
        return self.transform_info



def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, '..', 'Util', 'config.yaml')
    yaml_path = os.path.normpath(yaml_path)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(num_class=config['num_class']).to(device)

    graph = Draw_Graph(model=model, save_path=config['save_path'], patience=config['patience'])

    transform_info = model.return_transform_info()
    train_loader, validation_loader, test_loader = anomalydetection_data_loader(config['load_path'],
                                                                              config['batch_size'], transform_info)

    train_model(device=device, model=model, epochs=config['epoch'], patience=config['patience'], train_loader=train_loader,
                val_loader=validation_loader, test_loader=test_loader, lr=0.001, graph=graph)

if __name__ == "__main__":
    main()
