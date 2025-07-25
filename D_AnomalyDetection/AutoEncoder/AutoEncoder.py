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

    def create_network(self, resize_size):
        #256, 256, 3을 기준으로 계산
        row = resize_size[0]
        col = resize_size[1]
        input_channel = 3
        channel = 4

        # 가로, 세로 1/2
        self.network.append(nn.Conv2d(input_channel, channel * 2, kernel_size=4, stride=2, padding=1))  # 1. (128, 128, 8)
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        # 가로, 세로 그대로
        self.network.append(nn.Conv2d(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1))  # 2. (128, 128, 8)
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        # 채널 증폭 및 이미지 크기 계산
        row /= 2
        col /= 2
        channel *= 2
        while row > 20 and col > 20 :

            #가로, 세로 /2
            self.network.append(nn.Conv2d(channel, channel * 2, kernel_size=4, stride=2, padding=1)) # 3. (64, 64, 16) -> 5. (32, 32, 32) -> 7. (16, 16, 64)
            self.network.append(nn.LeakyReLU(0.2, inplace=True))

            #가로, 세로 그대로
            self.network.append(nn.Conv2d(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1)) # 4. (64, 64, 16) -> 6. (32, 32, 32) -> 8. (16, 16, 64)
            self.network.append(nn.LeakyReLU(0.2, inplace=True))

            # 채널 증폭 및 이미지 크기 계산
            row /= 2
            col /= 2
            channel *= 2

        self.network.append(nn.Conv2d(channel, 100, kernel_size=10, stride=1, padding=0)) # 9. (7, 7, 100)

        self.network.append(nn.ConvTranspose2d(100, channel, kernel_size=10, stride=1, padding=0)) # 8. (16, 16, 64)
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        while row == resize_size[0] / 2 and col == resize_size[1] / 2 :
            # 가로, 세로 *2
            self.network.append(nn.ConvTranspose2d(channel, int(channel/2), kernel_size=4, stride=2, padding=1)) # 9. (32, 32, 32), # 11. (64, 64, 16), # 12. (128, 128, 8)
            self.network.append(nn.LeakyReLU(0.2, inplace=True))

            # 가로, 세로 그대로
            self.network.append(nn.ConvTranspose2d(int(channel/2), int(channel/2), kernel_size=3, stride=1, padding=1)) #10. (32, 32, 32), # 11. (64, 64, 12), # 13. (128, 128, 8)
            self.network.append(nn.LeakyReLU(0.2, inplace=True))

            row *= 2
            col *= 2
            channel /= 2

        self.network.append(nn.ConvTranspose2d(channel, 3, kernel_size=4, stride=2, padding=1))  # 14. (256, 256, 3)
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
