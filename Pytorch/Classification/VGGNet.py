def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    return model


def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    return model


class VGGNet(nn.Module):
    def __init__(self, dim=64, num_classes=10):
        super(VGGNet, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, dim),
            conv_2_block(dim, dim*2),
            conv_3_block(dim*2, dim*4),
            conv_3_block(dim*4, dim*8),
            conv_3_block(dim*8, dim*8),
        )
        # fully connected + ReLU
        self.fc_layer = nn.Sequential(
            nn.Linear(dim*8*1*1, 4096),
            #1*1은 이미지 크기에 따라 변경(1*1은 32*32기준)
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = VGGNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
