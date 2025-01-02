import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import random
import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_path = 'D:/ImageNet/ImageNet/ILSVRC/Data/CLS-LOC/train'

dataset = ImageFolder(root=data_path, transform=transform)
train_indices = random.sample(range(len(dataset)), int(0.8 * len(dataset)))
valid_indices = [x for x in range(len(dataset)) if x not in train_indices]
train_dataset = Subset(dataset, train_indices)
valid_dataset = Subset(dataset, valid_indices)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=1, padding=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        return self.features(x)


teacher = Teacher().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-4)

num_epochs = 10
for epoch in range(num_epochs):
    teacher.train()
    train_loss = 0
    for images, _ in train_dataloader:
        images = images.to(device)
        optimizer.zero_grad()
        outputs = teacher(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    teacher.eval()
    valid_loss = 0
    with torch.no_grad():
        for images, _ in valid_dataloader:
            images = images.to(device)
            outputs = teacher(images)
            loss = criterion(outputs, images)
            valid_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_dataloader):.4f}, Valid Loss: {valid_loss / len(valid_dataloader):.4f}")

torch.save(teacher.state_dict(), 'Teacher_model5000.pth')
