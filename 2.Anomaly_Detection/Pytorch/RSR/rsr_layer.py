import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score

# RSR Layer implementation
class RSRLayer(nn.Module):
    def __init__(self, d:int, D: int):
        super().__init__()
        self.d = d
        self.D = D
        self.A = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(d, D)))

    def forward(self, z):
        z_hat = self.A @ z.view(z.size(0), self.D, 1)
        return z_hat.squeeze(2)

# RSR Loss implementation
class RSRLoss(nn.Module):
    def __init__(self, lambda1, lambda2, d, D):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.d = d
        self.D = D
        self.register_buffer("Id", torch.eye(d))

    def forward(self, z, A):
        z_hat = A @ z.view(z.size(0), self.D, 1)
        AtAz = (A.T @ z_hat).squeeze(2)
        term1 = torch.sum(torch.norm(z - AtAz, p=2))
        term2 = torch.norm(A @ A.T - self.Id, p=2) ** 2
        return self.lambda1 * term1 + self.lambda2 * term2

# L2,p Loss implementation
class L2p_Loss(nn.Module):
    def __init__(self, p=1.0):
        super().__init__()
        self.p = p

    def forward(self, y_hat, y):
        return torch.sum(torch.pow(torch.norm(y - y_hat, p=2), self.p))

# RSR Autoencoder implementation
class RSRAutoEncoder(nn.Module):
    def __init__(self, input_dim, d, D):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 4, D)
        )
        self.rsr = RSRLayer(d, D)
        self.decoder = nn.Sequential(
            nn.Linear(d, D),
            nn.LeakyReLU(),
            nn.Linear(D, input_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 2, input_dim)
        )

    def forward(self, x):
        enc = self.encoder(x)
        latent = self.rsr(enc)
        dec = self.decoder(latent)
        return enc, dec, latent, self.rsr.A

# Custom Dataset
class RSRDs(torch.utils.data.Dataset):
    def __init__(self, target_class, other_classes, n_examples_per_other):
        super().__init__()
        self.mnist = MNIST("..", download=True, transform=ToTensor())
        self.target_indices = (self.mnist.targets == target_class).nonzero().flatten()
        other = []
        for other_class in other_classes:
            other.extend((self.mnist.targets == other_class).nonzero().flatten()[:n_examples_per_other])
        self.other_indices = torch.tensor(other)
        self.all_indices = torch.cat([self.other_indices, self.target_indices])

    def __getitem__(self, idx):
        actual_idx = self.all_indices[idx].item()
        return self.mnist[actual_idx]

    def __len__(self):
        return self.all_indices.size(0)

# RSR Autoencoder Lightning Module
class RSRAE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.ae = RSRAutoEncoder(self.hparams['input_dim'], self.hparams['d'], self.hparams['D'])
        self.reconstruction_loss = L2p_Loss(p=1.0)
        self.rsr_loss = RSRLoss(self.hparams['lambda1'], self.hparams['lambda2'], self.hparams['d'], self.hparams['D'])

    def forward(self, x):
        return self.ae(x)

    def training_step(self, batch, batch_idx):
        X, _ = batch
        x = X.view(X.size(0), -1)
        enc, dec, latent, A = self.ae(x)
        rec_loss = self.reconstruction_loss(torch.sigmoid(dec), x)
        rsr_loss = self.rsr_loss(enc, A)
        loss = rec_loss + rsr_loss
        self.log("reconstruction_loss", rec_loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        self.log("rsr_loss", rsr_loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        return {"loss": loss}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.hparams['lr'], epochs=self.hparams['epochs'], steps_per_epoch=self.hparams['steps_per_epoch'])
        return [opt], [{"scheduler": scheduler, "interval": "step"}]

# Main execution
if __name__ == "__main__":
    pl.seed_everything(666)

    train_ds = RSRDs(target_class=4, other_classes=(0, 1, 2, 8), n_examples_per_other=100)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)

    val_ds = RSRDs(target_class=4, other_classes=(0, 1, 2, 8), n_examples_per_other=20)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

    hparams = dict(
        d=32, D=1024, input_dim=28*28,
        lr=0.01,
        epochs=10000, steps_per_epoch=len(train_dl),
        lambda1=1.0, lambda2=1.0,
    )

    model = RSRAE(hparams)
    trainer = pl.Trainer(max_epochs=hparams['epochs'], accelerator="gpu", devices=1)
    trainer.fit(model, train_dl)


    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dl:
            X, y = batch
            x = X.view(X.size(0), -1)
            _, x_hat, _, _ = model(x)
            rec_error = torch.norm(x - torch.sigmoid(x_hat), dim=1)
            preds = (rec_error > rec_error.mean()).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend((y == 4).int().cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {accuracy:.4f}")
