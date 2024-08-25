import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import optuna


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
    def __init__(self, alpha=1.0, num_classes=10):
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


def objective(trial):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alpha = trial.suggest_float("alpha", 0.5, 1.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    model = MobileNetV1(alpha=alpha).to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

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

    accuracy = 100 * correct / total
    return accuracy


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value}")
    print("  Best hyperparameters: ", trial.params)


if __name__ == '__main__':
    main()
