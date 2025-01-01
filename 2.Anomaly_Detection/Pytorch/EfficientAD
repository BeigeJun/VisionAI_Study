import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class PDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 4, padding=3)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 4, padding=3)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 384, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class StudentTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher = PDN()
        self.student = PDN()

    def forward(self, x):
        with torch.no_grad():
            teacher_out = self.teacher(x)
        student_out = self.student(x)
        return teacher_out, student_out


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(512, 64, 8)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 512, 8),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=3),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 384, 4, stride=2, padding=3)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.images = []
        self.labels = []

        if is_train:
            ok_folder = os.path.join(root_dir, 'Train', 'OK')
            self._add_images(ok_folder, 0)
        else:
            ok_folder = os.path.join(root_dir, 'Test', 'OK')
            ng_folder = os.path.join(root_dir, 'Test', 'NG')
            self._add_images(ok_folder, 0)
            self._add_images(ng_folder, 1)

    def _add_images(self, folder, label):
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            self.images.append(img_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(data_dir, transform=transform, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = CustomDataset(data_dir, transform=transform, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


def hard_feature_loss(teacher_out, student_out, p_hard=0.999):
    diff = (teacher_out - student_out) ** 2
    d_hard = torch.quantile(diff, p_hard)
    return torch.mean(diff[diff >= d_hard])


def pretraining_penalty(student, pretraining_image):
    student_out = student(pretraining_image)
    return torch.mean(student_out ** 2)


def autoencoder_loss(teacher_out, autoencoder_out):
    return F.mse_loss(teacher_out, autoencoder_out)


def student_autoencoder_loss(autoencoder_out, student_autoencoder_out):
    return F.mse_loss(autoencoder_out, student_autoencoder_out)


class ImageNetDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.image_paths = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            self.image_paths.extend([os.path.join(class_dir, img) for img in os.listdir(class_dir)])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)


def get_random_pretraining_image(batch_size, imagenet_dir):
    dataset = ImageNetDataset(imagenet_dir)
    indices = random.sample(range(len(dataset)), batch_size)
    batch = torch.stack([dataset[i] for i in indices])
    return batch


def train_efficientad(train_loader, val_loader, num_epochs, device, imagenet_dir):
    student_teacher = StudentTeacher().to(device)
    autoencoder = Autoencoder().to(device)
    optimizer = torch.optim.Adam([
        {'params': student_teacher.parameters()},
        {'params': autoencoder.parameters()}
    ], lr=1e-4)

    for epoch in range(num_epochs):
        student_teacher.train()
        autoencoder.train()
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            teacher_out, student_out = student_teacher(batch)
            st_loss = hard_feature_loss(teacher_out, student_out)

            pretraining_image = get_random_pretraining_image(batch.size(0), imagenet_dir).to(device)
            pt_loss = pretraining_penalty(student_teacher.student, pretraining_image)

            ae_out = autoencoder(batch)
            ae_loss = autoencoder_loss(teacher_out, ae_out)

            student_ae_out = student_teacher.student(batch)[:, 384:]  # 마지막 384 채널
            stae_loss = student_autoencoder_loss(ae_out, student_ae_out)

            total_loss = st_loss + pt_loss + ae_loss + stae_loss
            total_loss.backward()
            optimizer.step()

        validate(student_teacher, autoencoder, val_loader, device)

    return student_teacher, autoencoder


def validate(student_teacher, autoencoder, val_loader, device):
    student_teacher.eval()
    autoencoder.eval()
    total_loss = 0
    with torch.no_grad():
        for batch, _ in val_loader:
            batch = batch.to(device)
            teacher_out, student_out = student_teacher(batch)
            ae_out = autoencoder(batch)

            st_loss = hard_feature_loss(teacher_out, student_out)
            ae_loss = autoencoder_loss(teacher_out, ae_out)
            stae_loss = student_autoencoder_loss(ae_out, student_out[:, 384:])

            total_loss += st_loss + ae_loss + stae_loss

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")


def detect_anomalies(student_teacher, autoencoder, test_image):
    teacher_out, student_out = student_teacher(test_image)
    ae_out = autoencoder(test_image)

    local_map = torch.mean((teacher_out - student_out[:, :384]) ** 2, dim=1)
    global_map = torch.mean((ae_out - student_out[:, 384:]) ** 2, dim=1)

    combined_map = 0.5 * normalize_map(local_map) + 0.5 * normalize_map(global_map)
    return combined_map


def normalize_map(anomaly_map):
    q_a = torch.quantile(anomaly_map, 0.9)
    q_b = torch.quantile(anomaly_map, 0.995)
    return 0.1 * (anomaly_map - q_a) / (q_b - q_a)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "D:/Anomaly/Use/bottle_dataset"
    imagenet_dir = "D:/ImageNet/ImageNet/ILSVRC/Data/CLS-LOC/train"
    batch_size = 1
    num_epochs = 10

    train_loader, val_loader = get_data_loaders(data_dir, batch_size)

    student_teacher, autoencoder = train_efficientad(train_loader, val_loader, num_epochs, device, imagenet_dir)

    torch.save(student_teacher.state_dict(), "student_teacher.pth")
    torch.save(autoencoder.state_dict(), "autoencoder.pth")


if __name__ == "__main__":
    print("start")
    main()
