import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def conv_1_block(in_dim, out_dim, activation, stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride),
        nn.BatchNorm2d(out_dim),
        activation,
    )
    return model


def conv_3_block(in_dim, out_dim, activation, stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_dim),
        activation,
    )
    return model


class BottleNeck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, activation, down=False):
        super(BottleNeck, self).__init__()
        self.down = down
        if self.down:
            self.layer = nn.Sequential(
                conv_1_block(in_dim, mid_dim, activation, stride=2),
                conv_3_block(mid_dim, mid_dim, activation, stride=1),
                conv_1_block(mid_dim, out_dim, activation, stride=1),
            )
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2)
        else:
            self.layer = nn.Sequential(
                conv_1_block(in_dim, mid_dim, activation, stride=1),
                conv_3_block(mid_dim, mid_dim, activation, stride=1),
                conv_1_block(mid_dim, out_dim, activation, stride=1),
            )
        self.dim_equalizer = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out = out + downsample
        else:
            out = self.layer(x)
            if x.size() != out.size():
                x = self.dim_equalizer(x)
            out = out + x
        return out

class ResNet(nn.Module):
    def __init__(self, base_dim=64, num_classes=10):
        super(ResNet, self).__init__()
        self.activation = nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, base_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(),
        )
        self.layer_2 = nn.Sequential(
            BottleNeck(base_dim, base_dim, base_dim * 4, self.activation),
            BottleNeck(base_dim * 4, base_dim, base_dim * 4, self.activation),
            BottleNeck(base_dim * 4, base_dim, base_dim * 4, self.activation, down=True),
        )
        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim * 4, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation, down=True),
        )
        self.layer_4 = nn.Sequential(
            BottleNeck(base_dim * 8, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation, down=True),
        )
        self.layer_5 = nn.Sequential(
            BottleNeck(base_dim * 16, base_dim * 8, base_dim * 32, self.activation),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.activation),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.activation),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layer = nn.Linear(base_dim * 32, num_classes)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc_layer(out)
        return out

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ResNet().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Finished Training")
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%")
