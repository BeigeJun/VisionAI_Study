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
from tqdm import tqdm

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader

class DeepSVDDNetwork(nn.Module):
    def __init__(self, z_dim=32):
        super(DeepSVDDNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(64 * 64 * 64, z_dim)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bn3(self.conv3(x))), 2)
        x = x.view(x.size(0), -1)
        return self.fc1(x)


class TrainerDeepSVDD:
    def __init__(self, network, dataloader_train, device):
        self.network = network.to(device)
        self.dataloader_train = dataloader_train
        self.device = device

    def train(self, num_epochs=50, lr=1e-5):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        c = torch.randn(self.network.fc1.out_features).to(self.device)

        total_iterations = num_epochs * len(self.dataloader_train)
        pbar = tqdm(total=total_iterations, desc="Tra ning Progress")

        for epoch in range(num_epochs):
            total_loss = 0.0
            for images, _ in self.dataloader_train:
                images = images.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                outputs = self.network(images)
                loss = torch.mean(torch.sum((outputs - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                pbar.update(1)
                pbar.set_postfix({"Epoch": epoch + 1, "Loss": f"{total_loss / (epoch * len(self.dataloader_train) + len(images)):.4f}"})

        pbar.close()
        return c

def evaluate(network, c, dataloader_test, device):
    network.eval()
    scores = []
    labels = []
    correct = 0
    total = 0
    all_images = []
    all_predictions = []

    with torch.no_grad():
        for images, label in dataloader_test:
            images = images.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            outputs = network(images)
            score = torch.sum((outputs - c) ** 2, dim=1)
            scores.extend(score.cpu().numpy())
            labels.extend(label.cpu().numpy())

            threshold = torch.median(score)
            predictions = (score > threshold).long()
            correct += (predictions == label).sum().item()
            total += label.size(0)

            all_images.append(images.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    scores = np.array(scores)
    labels = np.array(labels)
    all_images = np.concatenate(all_images, axis=0)
    all_predictions = np.array(all_predictions)

    accuracy = correct / total
    auc_score = roc_auc_score(labels, scores)

    print(f"ROC AUC Score: {auc_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    ok_mask = labels == 0
    ng_mask = labels == 1
    ok_accuracy = ((scores[ok_mask] <= np.median(scores)).sum()) / ok_mask.sum()
    ng_accuracy = ((scores[ng_mask] > np.median(scores)).sum()) / ng_mask.sum()

    print(f"Accuracy for OK (Normal) samples: {ok_accuracy:.4f}")
    print(f"Accuracy for NG (Anomaly) samples: {ng_accuracy:.4f}")

    return auc_score, accuracy, ok_accuracy, ng_accuracy, all_images, labels, scores, all_predictions

def visualize_results(images, labels, scores, predicted_labels, batch_size=25):
    mean = np.array([0.5])  # 그레이스케일 이미지이므로 단일 값
    std = np.array([0.5])
    images = std * images.transpose(0, 2, 3, 1) + mean
    images = np.clip(images, 0, 1)

    num_images = len(images)
    if num_images == 0 or batch_size == 0:
        print("No images to visualize or invalid batch size.")
        return

    num_batches = max(1, (num_images + batch_size - 1) // batch_size)

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_images)

        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
        for i, ax in enumerate(axes.flat):
            idx = start_idx + i
            if idx < end_idx:
                ax.imshow(images[idx].squeeze(), cmap='gray')
                ax.axis('off')
                color = 'green' if predicted_labels[idx] == labels[idx] else 'red'
                ax.set_title(f'Pred: {predicted_labels[idx]}, True: {labels[idx]}', color=color)
            else:
                ax.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    train_dir = "D:/Anomaly/Use/bottle_dataset/Train"
    test_dir = "D:/Anomaly/Use/bottle_dataset/Test"
    batch_size = 32
    img_size = 512
    latent_dim = 32
    num_epochs = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloader_train, dataloader_test = get_custom_dataloader(train_dir, test_dir, batch_size=batch_size,
                                                              img_size=img_size)

    network = DeepSVDDNetwork(z_dim=latent_dim).to(device)
    trainer = TrainerDeepSVDD(network, dataloader_train, device)

    print("Training Deep SVDD...")
    c_center = trainer.train(num_epochs=num_epochs)

    print("Evaluating Deep SVDD...")
    auc_score, accuracy, ok_accuracy, ng_accuracy, images, labels, scores, predicted_labels = evaluate(network, c_center, dataloader_test, device)

    print("Visualizing results...")
    visualize_results(images, labels, scores, predicted_labels)
