import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_train=True):
        self.image_dir = image_dir
        self.is_train = is_train
        self.transform = transform

        if self.is_train:
            self.image_paths = [os.path.join(image_dir, 'OK', fname) for fname in
                                os.listdir(os.path.join(image_dir, 'OK')) if fname.endswith(('.png', '.jpg', '.jpeg'))]
        else:
            self.image_paths = []
            for label in ['OK', 'NG']:
                self.image_paths.extend(
                    [os.path.join(image_dir, label, fname) for fname in os.listdir(os.path.join(image_dir, label)) if
                     fname.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")
        if self.is_train:
            label = 0
        else:
            label = 0 if 'OK' in img_path else 1
        if self.transform:
            image = self.transform(image)
        return image, label


def get_custom_dataloader(train_dir, test_dir, batch_size, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = CustomImageDataset(train_dir, transform=transform, is_train=True)
    test_dataset = CustomImageDataset(test_dir, transform=transform, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class DeepSVDDNetwork(nn.Module):
    def __init__(self, z_dim=32):
        super(DeepSVDDNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 56 * 56, z_dim)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bn2(self.conv2(x))), 2)
        x = x.view(x.size(0), -1)
        return self.fc1(x)


class TrainerDeepSVDD:
    def __init__(self, network, dataloader_train, device):
        self.network = network.to(device)
        self.dataloader_train = dataloader_train
        self.device = device

    def train(self, num_epochs=50, lr=1e-3):
        c = torch.randn(self.network.fc1.out_features).to(self.device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        for epoch in range(num_epochs):
            total_loss = 0.0
            for images, _ in self.dataloader_train:
                images = images.to(self.device)
                optimizer.zero_grad()
                outputs = self.network(images)
                loss = torch.mean(torch.sum((outputs - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(
                f"Training Deep SVDD... Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(self.dataloader_train):.4f}")

        return c


def evaluate(network, c, dataloader_test, device):
    network.eval()
    scores = []
    labels = []
    images = []

    with torch.no_grad():
        for image, label in dataloader_test:
            image = image.to(device)
            outputs = network(image)
            score = torch.sum((outputs - c) ** 2, dim=1).cpu().numpy()
            scores.extend(score)
            labels.extend(label.numpy())
            images.extend(image.cpu().numpy())

    auc_score = roc_auc_score(labels, scores)
    print(f"ROC AUC Score: {auc_score:.4f}")

    show_results(images, labels, scores)

    return auc_score


def show_results(images, labels, scores):
    threshold = np.mean(scores)
    predicted_labels = (np.array(scores) > threshold).astype(int)

    for i in range(0, len(images), 10):
        batch_images = images[i:i + 10]
        batch_labels = labels[i:i + 10]
        batch_predictions = predicted_labels[i:i + 10]

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        for j, (img, label, pred) in enumerate(zip(batch_images, batch_labels, batch_predictions)):
            axes[j].imshow(img.squeeze(), cmap='gray')
            axes[j].set_title(f"Label: {label}, Pre: {pred}")
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    train_dir = "C:/Users/wns20/PycharmProjects/VisionAI_Study/2.Anomaly_Detection/ImageData/Train"
    test_dir = "C:/Users/wns20/PycharmProjects/VisionAI_Study/2.Anomaly_Detection/ImageData/Test"

    batch_size = 32
    img_size = 224
    latent_dim = 32
    num_epochs_train = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloader_train, dataloader_test = get_custom_dataloader(train_dir, test_dir, batch_size=batch_size,
                                                              img_size=img_size)

    network = DeepSVDDNetwork(z_dim=latent_dim).to(device)
    trainer = TrainerDeepSVDD(network, dataloader_train, device)

    print("Training Deep SVDD...")
    c_center = trainer.train(num_epochs=num_epochs_train)

    print("Evaluating Deep SVDD...")
    evaluate(network, c_center, dataloader_test, device)
