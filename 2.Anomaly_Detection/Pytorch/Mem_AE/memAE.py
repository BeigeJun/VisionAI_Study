import os
import pickle
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skimage.transform import resize


# Configuration
class Config:
    def __init__(self):
        self.dataset_dir = 'data'
        self.prepro_dir = 'preprocessed'
        self.num_instances = 1000
        self.image_height = 28
        self.image_width = 28
        self.image_channel_size = 1
        self.batch_size = 64
        self.num_epochs = 1
        self.learning_rate = 0.001
        self.num_dataloaders = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = 32
        self.num_memories = 10
        self.cls_loss_coef = 1.0
        self.entropy_loss_coef = 0.0
        self.condi_loss_coef = 1.0
        self.addressing = 'dense'


cfg = Config()


# Dataset
class MNIST_Dataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.load_dataset()

    def load_dataset(self):
        self.train_dataset = datasets.MNIST(root=self.cfg.dataset_dir, train=True, download=True,
                                            transform=self.transform)
        self.test_dataset = datasets.MNIST(root=self.cfg.dataset_dir, train=False, download=True,
                                           transform=self.transform)

        # Limit the number of instances if specified
        if self.cfg.num_instances > 0:
            self.train_dataset.data = self.train_dataset.data[:self.cfg.num_instances]
            self.train_dataset.targets = self.train_dataset.targets[:self.cfg.num_instances]


# Model
class ICVAE(nn.Module):
    def __init__(self, cfg):
        super(ICVAE, self).__init__()
        self.cfg = cfg

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256, cfg.latent_dim)
        self.fc_logvar = nn.Linear(256, cfg.latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

        # Memory
        self.memory = nn.Parameter(torch.randn(cfg.num_memories, cfg.latent_dim))

        # Classifier
        self.classifier = nn.Linear(cfg.latent_dim, cfg.num_memories)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        mem_weight = torch.softmax(self.classifier(z), dim=1)
        return {'rec_x': x_recon, 'mu': mu, 'logvar': logvar, 'z': z, 'mem_weight': mem_weight}


# Trainer
class Trainer:
    def __init__(self, cfg, model, optimizer, train_loader, test_loader):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = cfg.device

        self.rec_criterion = nn.MSELoss(reduction='sum')
        self.cls_criterion = nn.CrossEntropyLoss(reduction='mean')

    def train(self):
        self.model.train()
        for epoch in range(self.cfg.num_epochs):
            train_loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                result = self.model(data)

                rec_loss = self.rec_criterion(result['rec_x'], data)
                kl_loss = -0.5 * torch.sum(1 + result['logvar'] - result['mu'].pow(2) - result['logvar'].exp())
                cls_loss = self.cls_criterion(result['mem_weight'], target)

                loss = rec_loss + kl_loss + self.cfg.cls_loss_coef * cls_loss
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            print(f'Epoch {epoch + 1}, Train loss: {train_loss / len(self.train_loader.dataset):.4f}')

            if (epoch + 1) % 10 == 0:
                self.test()

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                result = self.model(data)
                test_loss += self.rec_criterion(result['rec_x'], data).item()
                pred = result['mem_weight'].argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        print(f'Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')


# Main execution
def main():
    dataset = MNIST_Dataset(cfg)
    train_loader = DataLoader(dataset.train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_dataloaders)
    test_loader = DataLoader(dataset.test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_dataloaders)

    model = ICVAE(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    trainer = Trainer(cfg, model, optimizer, train_loader, test_loader)
    trainer.train()


if __name__ == '__main__':
    main()
