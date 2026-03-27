import os
import torch
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.configs = [
            [100, 512, 4, 1, 0, False],
            [512, 256, 4, 2, 1, False],
            [256, 128, 4, 2, 1, False],
            [128, 64, 4, 2, 1, False],
            [64, 3, 4, 2, 1, False]]

        layers = []
        for i, (Input, Output, Kernel, Stride, Padding, Bias) in enumerate(self.configs):
            layers.append(nn.ConvTranspose2d(in_channels=Input, out_channels=Output, kernel_size=Kernel, stride=Stride,
                                    padding=Padding, bias=Bias))
            if i != len(self.configs) - 1:
                layers.append(nn.BatchNorm2d(Output))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())
        self.Model = nn.Sequential(*layers)

    def forward(self, x):
        return self.Model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.configs = [
            [3, 64, 4, 2, 1, False],
            [64, 128, 4, 2, 1, False],
            [128, 256, 4, 2, 1, False],
            [256, 512, 4, 2, 1, False],
            [512, 1, 4, 1, 0, False]]

        layers = []
        for i, (Input, Output, Kernel, Stride, Padding, Bias) in enumerate(self.configs):
            layers.append(nn.Conv2d(in_channels=Input, out_channels=Output, kernel_size=Kernel, stride=Stride,
                                    padding=Padding, bias=Bias))
            if i != len(self.configs) - 1:
                layers.append(nn.BatchNorm2d(Output))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Sigmoid())
        self.Model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.Model(x)
        return out.view(-1,1)


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0


def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


def train(model_generator, model_discriminator, opt_generator, opt_discriminator, train_dl, loss_func, Epochs):
    epoch_bar = tqdm(range(Epochs), desc="Epoch", unit="epoch")
    for _ in epoch_bar:
        total_g_loss = 0.0
        total_d_loss = 0.0
        num_batches = len(train_dl)

        for xb, _ in train_dl:

            ba_si = xb.size(0)
            xb = xb.to(device)

            yb_real = torch.full((ba_si, 1), 1.0, device=device)
            yb_fake = torch.full((ba_si, 1), 0.0, device=device)

            model_generator.zero_grad()
            z = torch.randn(ba_si, 100, 1, 1, device=device)
            out_gen = model_generator(z)
            out_dis = model_discriminator(out_gen)
            g_loss = loss_func(out_dis, yb_real)
            g_loss.backward()
            opt_generator.step()

            model_discriminator.zero_grad()
            out_dis_real = model_discriminator(xb)
            loss_real = loss_func(out_dis_real, yb_real)
            out_dis_fake = model_discriminator(out_gen.detach())
            loss_fake = loss_func(out_dis_fake, yb_fake)
            d_loss = (loss_real + loss_fake) / 2
            d_loss.backward()
            opt_discriminator.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

        avg_g_loss = total_g_loss / num_batches
        avg_d_loss = total_d_loss / num_batches

        epoch_bar.set_postfix(
            g_loss=f"{avg_g_loss:.4f}",
            d_loss=f"{avg_d_loss:.4f}"
        )

    return model_generator


def generate_images(generator, num_images, save_dir, device='cpu'):
    generator.eval()  # 평가 모드로 전환
    os.makedirs(save_dir, exist_ok=True)
    noise = torch.randn(num_images, 100, 1, 1, device=device)  # latent vector 생성
    with torch.no_grad():
        fake_images = generator(noise).detach().cpu()
    # 이미지 값 복원 (tanh 사용 시 -1~1 → 0~1로 변환)
    fake_images = (fake_images + 1) / 2
    for i in range(num_images):
        save_path = os.path.join(save_dir, f"generated_{i+1}.png")
        save_image(fake_images[i], save_path)


def main():
    model_generator = Generator().to(device)
    model_discriminator = Discriminator().to(device)

    model_generator.apply(init_weights)
    model_discriminator.apply(init_weights)

    opt_generator = optim.Adam(model_generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_discriminator = optim.Adam(model_discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_ds = CustomImageDataset(root_dir='D:/Image_Data/OneLabel/images', transform=transform)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

    # import matplotlib.pyplot as plt
    # from torchvision.transforms.functional import to_pil_image
    # img, _ = train_ds[0]
    # plt.imshow(to_pil_image(0.5 * img + 0.5))
    # plt.show()

    loss_func = nn.BCELoss()
    Epochs = 10000

    path = 'D:/Model_Save/Gan'
    model_generator = train(model_generator, model_discriminator, opt_generator, opt_discriminator, train_dl, loss_func, Epochs)
    torch.save(model_generator.state_dict(), path + '/model/0/generator.pth')
    generate_images(model_generator, num_images=10, save_dir=path + "/image/0", device=device)
if __name__ == "__main__":
    main()
