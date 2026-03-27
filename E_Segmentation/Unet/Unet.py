import sys
import os
import yaml
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.append(root_path)
from E_Segmentation.Util.Util import *
from E_Segmentation.Util.Draw_Graph import *

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.enc1 = DoubleConv(3, 64); self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256); self.enc4 = DoubleConv(256, 512)
        self.bottom = DoubleConv(512, 1024); self.pool = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2); self.dec1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.dec2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.dec3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.dec4 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x); p1 = self.pool(c1); c2 = self.enc2(p1); p2 = self.pool(c2)
        c3 = self.enc3(p2); p3 = self.pool(c3); c4 = self.enc4(p3); p4 = self.pool(c4)
        b = self.bottom(p4)
        d1 = self.dec1(torch.cat([self.up1(b), c4], 1))
        d2 = self.dec2(torch.cat([self.up2(d1), c3], 1))
        d3 = self.dec3(torch.cat([self.up3(d2), c2], 1))
        d4 = self.dec4(torch.cat([self.up4(d3), c1], 1))
        return self.out_conv(d4)
    

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, '..', 'Util', 'config.yaml')
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=config['num_class']).to(device)

    train_loader, val_loader, test_loader = segmentation_data_loader(
        config['load_path'], config['batch_size'], config['input_size']
    )

    graph = Draw_Graph(model=model, save_path=config['save_path'], patience=config['patience'])

    # train_model(
    #     device=device, model=model, train_loader=train_loader, 
    #     val_loader=val_loader, graph=graph, epochs=config['epoch'], 
    #     lr=1e-4, patience=config['patience'], save_path=config['save_path']
    # )

    model.load_state_dict(torch.load(os.path.join(config['save_path'], "Best_Accuracy_Validation.pth")))
    visualize_random_10(model, device, config['load_path'], config['input_size'])

if __name__ == "__main__":
    main()