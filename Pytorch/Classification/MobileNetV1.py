class DepSepConv(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, stride=stride, padding=1, groups=in_dim, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU6(),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU6()
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetV1(nn.Module):
    def __init__(self, alpha, num_classes=10):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(32 * alpha), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(32 * alpha)),
            nn.ReLU(inplace=True),
        )
        self.conv2 = DepSepConv(int(32 * alpha), int(64 * alpha))

        self.conv3 = nn.Sequential(
            DepSepConv(int(64 * alpha), int(128 * alpha), stride=2),
            DepSepConv(int(128 * alpha), int(128 * alpha))
        )
        self.conv4 = nn.Sequential(
            DepSepConv(int(128 * alpha), int(256 * alpha), stride=2),
            DepSepConv(int(256 * alpha), int(256 * alpha))
        )
        self.conv5 = nn.Sequential(
            DepSepConv(int(256 * alpha), int(512 * alpha), stride=2),
            DepSepConv(int(512 * alpha), int(512 * alpha), stride=1),
            DepSepConv(int(512 * alpha), int(512 * alpha), stride=1),
            DepSepConv(int(512 * alpha), int(512 * alpha), stride=1),
            DepSepConv(int(512 * alpha), int(512 * alpha), stride=1),
            DepSepConv(int(512 * alpha), int(512 * alpha), stride=1),
        )
        # self.conv5 = nn.Sequential(
        #     DepSepConv(int(256 * alpha), int(512 * alpha), stride=2),
        #     *[DepSepConv(int(512 * alpha), int(512 * alpha)) for _ in range(5)]
        # )
        self.conv6 = nn.Sequential(
            DepSepConv(int(512 * alpha), int(1024 * alpha), stride=2),
            DepSepConv(int(1024 * alpha), int(1024 * alpha))
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(1024 * alpha), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
