import numpy as np
import easydict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class CustomDataset(data.Dataset):
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

def get_data_loaders(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(args.data_dir, transform=transform, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_dataset = CustomDataset(args.data_dir, transform=transform, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

class DeepSVDD_network(nn.Module):
    def __init__(self, z_dim=32):
        super(DeepSVDD_network, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 128, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(128, 64, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(64 * 56 * 56, z_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)


class pretrain_autoencoder(nn.Module):
    def __init__(self, z_dim=64):
        super(pretrain_autoencoder, self).__init__()
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 128, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(128, 64, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(64 * 56 * 56, z_dim, bias=False)

        self.deconv1 = nn.ConvTranspose2d(z_dim, 64, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(64, 128, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(128, 3, 5, bias=False, padding=2)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

    def decoder(self, x):
        x = x.view(x.size(0), self.z_dim, 1, 1)
        x = F.interpolate(F.leaky_relu(x), scale_factor=56)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        return torch.sigmoid(x)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class TrainerDeepSVDD:
    def __init__(self, args, data_loader, device):
        self.args = args
        self.train_loader = data_loader
        self.device = device

    def pretrain(self):
        ae = pretrain_autoencoder(self.args.latent_dim).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = torch.optim.Adam(ae.parameters(), lr=self.args.lr_ae, weight_decay=self.args.weight_decay_ae)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)

        ae.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                x_hat = ae(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()

                total_loss += reconst_loss.item()
            scheduler.step()
            print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(
                epoch, total_loss / len(self.train_loader)))
        self.save_weights_for_DeepSVDD(ae, self.train_loader)

    def train(self):
        net = DeepSVDD_network().to(self.device)
        if self.args.pretrain:
            state_dict = torch.load('pretrained_parameters.pth')
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            net.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)

        net.train()
        pbar = tqdm(range(self.args.num_epochs), desc="Training Deep SVDD")
        for epoch in pbar:
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)
                optimizer.zero_grad()
                z = net(x)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            pbar.set_postfix({'Loss': f'{total_loss / len(self.train_loader):.10f}'})

        self.net = net
        self.c = c
        return self.net, self.c

    def save_weights_for_DeepSVDD(self, model, dataloader):
        c = self.set_c(model, dataloader)
        net = DeepSVDD_network(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, 'pretrained_parameters.pth')

    def set_c(self, model, dataloader, eps=0.1):
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encoder(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

def eval(net, c, dataloader, device):
    scores = []
    labels = []
    images = []
    net.eval()
    print('Testing...')
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            z = net(x)
            score = torch.sum((z - c) ** 2, dim=1)
            scores.append(score.detach().cpu())
            labels.append(y.cpu())
            images.append(x.cpu())

    labels = torch.cat(labels).numpy()
    scores = torch.cat(scores).numpy()
    images = torch.cat(images).numpy()

    threshold = np.percentile(scores, 90)
    predicted_labels = (scores > threshold).astype(int)
    correct = np.sum(predicted_labels == labels)
    incorrect = np.sum(predicted_labels != labels)

    print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels, scores) * 100))
    print('Basic score: {:.2f}'.format(correct/(correct+incorrect)*100))
    print(f'맞은 개수: {correct}, 틀린 개수: {incorrect}')

    return labels, scores, images, predicted_labels

def visualize_results(images, labels, scores, predicted_labels, batch_size=25):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
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
                ax.imshow(images[idx])
                ax.axis('off')
                color = 'green' if predicted_labels[idx] == labels[idx] else 'red'
                ax.set_title(f'Pred: {predicted_labels[idx]}, True: {labels[idx]}', color=color)
            else:
                ax.axis('off')
        plt.tight_layout()
        plt.show()


def visualize_distribution(net, c, dataloader, device):
    net.eval()
    features = []
    labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            z = net(x)
            features.append(z.cpu().numpy())
            labels.append(y.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    c_2d = pca.transform(c.cpu().numpy().reshape(1, -1))

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter)

    plt.scatter(c_2d[0, 0], c_2d[0, 1], c='yellow', s=200, marker='*', edgecolors='black', linewidth=1.5,
                label='Center')

    distances = np.sum((features - c.cpu().numpy()) ** 2, axis=1)
    radius = np.percentile(distances, 90)

    radius_2d = np.sqrt(radius) * np.mean(np.std(features_2d, axis=0))
    circle = plt.Circle((c_2d[0, 0], c_2d[0, 1]), radius_2d, fill=False, color='green', linestyle='--',
                        label='Radius (90th percentile)')
    plt.gca().add_artist(circle)

    plt.legend()
    plt.title('DeepSVDD Feature Distribution')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    plt.xlim(c_2d[0, 0] - radius_2d * 1.5, c_2d[0, 0] + radius_2d * 1.5)
    plt.ylim(c_2d[0, 1] - radius_2d * 1.5, c_2d[0, 1] + radius_2d * 1.5)

    plt.show()


if __name__ == '__main__':
    args = easydict.EasyDict({
        'num_epochs': 1000,
        'num_epochs_ae': 1000,
        'lr': 1e-3,
        'lr_ae': 1e-2,
        'weight_decay': 5e-7,
        'weight_decay_ae': 5e-3,
        'lr_milestones': [50],
        'batch_size': 8,
        'pretrain': True,
        'latent_dim': 32,
        'data_dir': 'D:/Anomaly/Use/metalnut_dataset'
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader_train, dataloader_test = get_data_loaders(args)
    deep_SVDD = TrainerDeepSVDD(args, dataloader_train, device)

    if args.pretrain:
        deep_SVDD.pretrain()

    net, c = deep_SVDD.train()

    labels, scores, images, predicted_labels = eval(net, c, dataloader_test, device)

    visualize_results(images, labels, scores, predicted_labels)

    visualize_distribution(net, c, dataloader_test, device)