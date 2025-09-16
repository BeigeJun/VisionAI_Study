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
        # 256, 256, 3을 기준으로 계산
        row = resize_size[0]
        col = resize_size[1]
        input_channel = 3
        channel = 4

        # 가로, 세로 1/2
        self.network.append(
            nn.Conv2d(input_channel, channel * 2, kernel_size=4, stride=2, padding=1))  # 1. (128, 128, 8)
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        # 가로, 세로 그대로
        self.network.append(nn.Conv2d(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1))  # 2. (128, 128, 8)
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        # 채널 증폭 및 이미지 크기 계산
        row /= 2
        col /= 2
        channel *= 2
        while row > 20 and col > 20:
            # 가로, 세로 /2
            self.network.append(nn.Conv2d(channel, channel * 2, kernel_size=4, stride=2,
                                          padding=1))  # 3. (64, 64, 16) -> 5. (32, 32, 32) -> 7. (16, 16, 64)
            self.network.append(nn.LeakyReLU(0.2, inplace=True))

            # 가로, 세로 그대로
            self.network.append(nn.Conv2d(channel * 2, channel * 2, kernel_size=3, stride=1,
                                          padding=1))  # 4. (64, 64, 16) -> 6. (32, 32, 32) -> 8. (16, 16, 64)
            self.network.append(nn.LeakyReLU(0.2, inplace=True))

            # 채널 증폭 및 이미지 크기 계산
            row /= 2
            col /= 2
            channel *= 2

        # 인코더 마지막 레이어 (채널 100, 크기 row - 9, col - 9)
        if int(row) < 10 or int(col) < 10:
            # 안정성을 위해 커널 사이즈 7로 축소
            kernel_size_enc = 7
        else:
            kernel_size_enc = 10

        self.network.append(nn.Conv2d(channel, 100, kernel_size=kernel_size_enc, stride=1, padding=0))  # 인코딩 레이어
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        # 디코더 시작
        self.network.append(nn.ConvTranspose2d(100, channel, kernel_size=kernel_size_enc, stride=1, padding=0))
        self.network.append(nn.LeakyReLU(0.2, inplace=True))

        # 디코더 업샘플링 조건, 주의: 루프 조건 수정 (or 사용)
        while int(row) != int(resize_size[0] / 2) or int(col) != int(resize_size[1] / 2):
            self.network.append(
                nn.ConvTranspose2d(int(channel), int(channel / 2), kernel_size=4, stride=2, padding=1))  # 업샘플링
            self.network.append(nn.LeakyReLU(0.2, inplace=True))

            self.network.append(
                nn.ConvTranspose2d(int(channel / 2), int(channel / 2), kernel_size=3, stride=1, padding=1))  # 채널 유지
            self.network.append(nn.LeakyReLU(0.2, inplace=True))

            row *= 2
            col *= 2
            channel = int(channel / 2)

        # 추가 ConvTranspose2d (채널 downscale)
        self.network.append(nn.ConvTranspose2d(channel, int(channel / 2), kernel_size=3, stride=1, padding=1))
        self.network.append(nn.LeakyReLU(0.2, inplace=True))
        channel = int(channel / 2)

        # 마지막 출력층 (채널 3)
        self.network.append(nn.ConvTranspose2d(channel, 3, kernel_size=4, stride=2, padding=1))  # (256, 256, 3)
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
