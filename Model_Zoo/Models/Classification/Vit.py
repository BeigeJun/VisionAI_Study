import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchsummary import summary
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

from torch import nn, Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> qkv b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size: int = 768, drop_p: float = 0., forward_expansion: int = 4,
                 forward_drop_p: float = 0., num_heads: int = 8, **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads=num_heads, dropout=drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, emb_size: int = 768, num_heads: int = 8, **kwargs):
        super().__init__(*[TransformerEncoderBlock(emb_size=emb_size, num_heads=num_heads, **kwargs)
                           for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )


class ViT(nn.Sequential):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768,
                 img_size: int = 224, depth: int = 12, n_classes: int = 1000, num_heads: int = 8, **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, num_heads=num_heads, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )


class ImageDataset:
    def __init__(self, data_dir='A', img_size=224, batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size

        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.val_test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_datasets(self):
        return (
            datasets.ImageFolder(os.path.join(self.data_dir, 'Train'), self.train_transform),
            datasets.ImageFolder(os.path.join(self.data_dir, 'Validation'), self.val_test_transform),
            datasets.ImageFolder(os.path.join(self.data_dir, 'Test'), self.val_test_transform)
        )

    def create_loaders(self):
        train, val, test = self.load_datasets()
        return (
            DataLoader(train, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(val, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True),
            DataLoader(test, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        )


class ViTTrainer:
    def __init__(self, model, device, num_classes):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.1)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.best_acc = 0.0

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return total_loss / total, correct / total

    def evaluate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / total, correct / total

    def save_model(self, path='best_vit.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = ImageDataset(data_dir='D:/Image_Data/Test', img_size=224, batch_size=64)
    train_loader, val_loader, test_loader = dataset.create_loaders()

    num_classes = len(train_loader.dataset.classes)
    vit_model = ViT(n_classes=num_classes)
    vit_model.to(device)

    print("\n" + "=" * 50)
    summary(vit_model, (3, 224, 224), device=device.type)
    print("=" * 50 + "\n")

    trainer = ViTTrainer(vit_model, device, num_classes)

    num_epochs = 100
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        trainer.scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trainer.save_model()
            print(f"New best model saved at epoch {epoch + 1} with val acc {val_acc:.4f}")

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print("-" * 50)

    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\nFinal Test Performance - Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
