import sys
import os
import yaml
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.append(root_path)
from E_Segmentation.Util.Util import *
from E_Segmentation.Util.Draw_Graph import *

import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        reduced_channels = max(1, int(in_channels * se_ratio))
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            Swish(),
            nn.Linear(reduced_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        se = self.squeeze(x).view(bs, c)
        se = self.excitation(se).view(bs, c, 1, 1)
        return x * se

class FusedMBConv(nn.Module):
    def __init__(self, in_put, out_put, kernel_size, stride, expansion, se_ratio=None):
        super().__init__()
        self.use_shortcut = (stride == 1) and (in_put == out_put)
        hidden_dim = in_put * expansion

        layers = []
        layers.append(nn.Conv2d(in_put, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(Swish())

        if se_ratio:
            layers.append(SEBlock(hidden_dim, se_ratio))

        layers.append(nn.Conv2d(hidden_dim, out_put, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_put))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.block(x)
        return self.block(x)

class MBconv(nn.Module):
    def __init__(self, in_put, out_put, kernel_size, stride, expansion, se_ratio=0.25):
        super().__init__()
        self.use_shortcut = (stride == 1) and (in_put == out_put)
        hidden_dim = in_put * expansion

        layers = []
        if expansion != 1:
            layers.append(nn.Conv2d(in_put, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(Swish())

        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(Swish())
        layers.append(SEBlock(hidden_dim, se_ratio))

        layers.append(nn.Conv2d(hidden_dim, out_put, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_put))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.block(x)
        return self.block(x)

class EfficientUNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            Swish()
        )
        
        self.stage1 = FusedMBConv(24, 24, 3, 1, 1)
        self.stage2 = FusedMBConv(24, 48, 3, 2, 4)
        self.stage3 = FusedMBConv(48, 64, 3, 2, 4)
        self.stage4 = MBconv(64, 128, 3, 2, 4, 0.25)
        self.stage5 = MBconv(128, 160, 3, 2, 6, 0.25)

        self.up1 = nn.ConvTranspose2d(160, 128, 2, 2)
        self.dec1 = nn.Sequential(nn.Conv2d(128 + 128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        
        self.up3 = nn.ConvTranspose2d(64, 48, 2, 2)
        self.dec3 = nn.Sequential(nn.Conv2d(48 + 48, 48, 3, 1, 1), nn.BatchNorm2d(48), nn.ReLU(True))
        
        self.up4 = nn.ConvTranspose2d(48, 24, 2, 2)
        self.dec4 = nn.Sequential(nn.Conv2d(24 + 24, 24, 3, 1, 1), nn.BatchNorm2d(24), nn.ReLU(True))
        
        self.final_up = nn.ConvTranspose2d(24, 24, 2, 2)
        self.out_conv = nn.Conv2d(24, num_classes, 1)

    def forward(self, x):
        s0 = self.stem(x)
        e1 = self.stage1(s0)
        e2 = self.stage2(e1)
        e3 = self.stage3(e2)
        e4 = self.stage4(e3)
        e5 = self.stage5(e4)
        
        d1 = self.dec1(torch.cat([self.up1(e5), e4], 1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], 1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], 1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], 1))
        
        return self.out_conv(self.final_up(d4))
    

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, '..', 'Util', 'config.yaml')
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientUNetV2(num_classes=config['num_class']).to(device)

    train_loader, val_loader, test_loader = segmentation_data_loader(
        config['load_path'], config['batch_size'], config['input_size']
    )

    graph = Draw_Graph(model=model, save_path=config['save_path'], patience=config['patience'])
    model.load_state_dict(torch.load(os.path.join(config['save_path'], "Best_Accuracy_Validation.pth")))
    # train_model(
    #     device=device, model=model, train_loader=train_loader, 
    #     val_loader=val_loader, graph=graph, epochs=config['epoch'], 
    #     lr=1e-4, patience=config['patience'], save_path=config['save_path']
    # )

    visualize_random_10(model, device, config['load_path'], config['input_size'])

if __name__ == "__main__":
    main()