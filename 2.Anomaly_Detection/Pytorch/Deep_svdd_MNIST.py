import numpy as np
import easydict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class MNIST_loader(data.Dataset):
    """Preprocessing을 포함한 dataloader를 구성"""

    def __init__(self, data, target, transform):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


def get_mnist(args, data_dir='C:/Users/wns20/PycharmProjects/VisionAI_Study/0.Data/MNIST'):
    """get dataloders"""
    # min, max values for each class after applying GCN (as the original implementation)
    min_max = [(-0.8826567065619495, 9.001545489292527),
               (-0.6661464580883915, 20.108062262467364),
               (-0.7820454743183202, 11.665100841080346),
               (-0.7645772083211267, 12.895051191467457),
               (-0.7253923114302238, 12.683235701611533),
               (-0.7698501867861425, 13.103278415430502),
               (-0.778418217980696, 10.457837397569108),
               (-0.7129780970522351, 12.057777597673047),
               (-0.8280402650205075, 10.581538445782988),
               (-0.7369959242164307, 10.697039838804978)]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: global_contrast_normalization(x)),
                                    transforms.Normalize([min_max[args.normal_class][0]],
                                                         [min_max[args.normal_class][1] \
                                                          - min_max[args.normal_class][0]])])
    train = datasets.MNIST(root=data_dir, train=True, download=True)
    test = datasets.MNIST(root=data_dir, train=False, download=True)

    x_train = train.data
    y_train = train.targets

    x_train = x_train[np.where(y_train == args.normal_class)]
    y_train = y_train[np.where(y_train == args.normal_class)]

    data_train = MNIST_loader(x_train, y_train, transform)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)

    x_test = test.data
    y_test = test.targets

    # Normal class인 경우 0으로 바꾸고, 나머지는 1로 변환 (정상 vs 비정상 class)
    y_test = np.where(y_test == args.normal_class, 0, 1)

    data_test = MNIST_loader(x_test, y_test, transform)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)
    return dataloader_train, dataloader_test


def global_contrast_normalization(x):
    """Apply global contrast normalization to tensor. """
    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean
    x_scale = torch.mean(torch.abs(x))
    x /= x_scale
    return x


class DeepSVDD_network(nn.Module):
    def __init__(self, z_dim=32):
        super(DeepSVDD_network, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)


class pretrain_autoencoder(nn.Module):
    def __init__(self, z_dim=32):
        super(pretrain_autoencoder, self).__init__()
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)

        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

    def decoder(self, x):
        x = x.view(x.size(0), int(self.z_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
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
        """ DeepSVDD 모델에서 사용할 가중치를 학습시키는 AutoEncoder 학습 단계"""
        ae = pretrain_autoencoder(self.args.latent_dim).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = torch.optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                                     weight_decay=self.args.weight_decay_ae)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.lr_milestones, gamma=0.1)

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

    def save_weights_for_DeepSVDD(self, model, dataloader):
        """학습된 AutoEncoder 가중치를 DeepSVDD모델에 Initialize해주는 함수"""
        c = self.set_c(model, dataloader)
        net = DeepSVDD_network(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, 'pretrained_parameters.pth')

    def set_c(self, model, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
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

    def train(self):
        """Deep SVDD model 학습"""
        net = DeepSVDD_network().to(self.device)

        if self.args.pretrain == True:
            state_dict = torch.load('pretrained_parameters.pth')
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            net.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr,
                                     weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.lr_milestones, gamma=0.1)

        net.train()
        for epoch in range(self.args.num_epochs):
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
            print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(
                epoch, total_loss / len(self.train_loader)))
        self.net = net
        self.c = c

        return self.net, self.c

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

def pretrain(self):
    """ DeepSVDD 모델에서 사용할 가중치를 학습시키는 AutoEncoder 학습 단계"""
    ae = pretrain_autoencoder(self.args.latent_dim).to(self.device)
    ae.apply(weights_init_normal)
    optimizer = torch.optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                                 weight_decay=self.args.weight_decay_ae)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=self.args.lr_milestones, gamma=0.1)

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


def save_weights_for_DeepSVDD(self, model, dataloader):
    """학습된 AutoEncoder 가중치를 DeepSVDD모델에 Initialize해주는 함수"""
    c = self.set_c(model, dataloader)
    net = DeepSVDD_network(self.args.latent_dim).to(self.device)
    state_dict = model.state_dict()
    net.load_state_dict(state_dict, strict=False)
    torch.save({'center': c.cpu().data.numpy().tolist(),
                'net_dict': net.state_dict()}, 'pretrained_parameters.pth')


def train(self):
    """Deep SVDD model 학습"""
    net = DeepSVDD_network().to(self.device)

    if self.args.pretrain == True:
        state_dict = torch.load('pretrained_parameters.pth')
        net.load_state_dict(state_dict['net_dict'])
        c = torch.Tensor(state_dict['center']).to(self.device)
    else:
        net.apply(weights_init_normal)
        c = torch.randn(self.args.latent_dim).to(self.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr,
                                 weight_decay=self.args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=self.args.lr_milestones, gamma=0.1)

    net.train()
    for epoch in range(self.args.num_epochs):
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
        print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(
            epoch, total_loss / len(self.train_loader)))
    self.net = net
    self.c = c

    return self.net, self.c


def eval(net, c, dataloader, device):
    """Testing the Deep SVDD model"""
    scores = []
    labels = []
    net.eval()
    print('Testing...')
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            z = net(x)
            score = torch.sum((z - c) ** 2, dim=1)
            scores.append(score.detach().cpu())
            labels.append(y.cpu())

    labels = torch.cat(labels).numpy()
    scores = torch.cat(scores).numpy()

    # ROC AUC 점수 계산
    roc_auc = roc_auc_score(labels, scores) * 100

    # 임계값 설정 (예: 점수의 중앙값)
    threshold = np.median(scores)

    # 예측 수행
    predictions = (scores > threshold).astype(int)

    # 정확도 계산
    accuracy = (predictions == labels).mean() * 100

    # 맞은 개수와 틀린 개수 계산
    correct = (predictions == labels).sum()
    incorrect = len(labels) - correct

    print(f'ROC AUC score: {roc_auc:.2f}')
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Correct predictions: {correct}')
    print(f'Incorrect predictions: {incorrect}')

    return labels, scores, predictions


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = easydict.EasyDict({
       'num_epochs':5000,
       'num_epochs_ae':5000,
       'lr':1e-3,
       'lr_ae':1e-3,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'batch_size':1024,
       'pretrain':True,
       'latent_dim':32,
       'normal_class':0
                })

if __name__ == '__main__':

    dataloader_train, dataloader_test = get_mnist(args)

    deep_SVDD = TrainerDeepSVDD(args, dataloader_train, device)

    if args.pretrain:
        deep_SVDD.pretrain()

    net, c = deep_SVDD.train()
    eval(net, c, dataloader_test, device)
