import sys
import os
import yaml
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.append(root_path)
from E_Segmentation.Util.Util import *
from E_Segmentation.Util.Draw_Graph import *

class ProjectionCut(nn.Module):
    def __init__(self, in_put, out_put, stride, expansion=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_put, out_channels=out_put*expansion, kernel_size=1, stride=stride,
                              bias=False)
        self.batch = nn.BatchNorm2d(out_put*expansion)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_put, out_put, stride=1, expansion=4):
        super().__init__()
        self.expansion = expansion

        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_put, out_channels=out_put, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_put),
            nn.ReLU()
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=out_put, out_channels=out_put, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_put),
            nn.ReLU()
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_put, out_channels=out_put*expansion, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_put*expansion)
        )

        if stride != 1 or in_put != out_put * expansion:
            self.short_cut = ProjectionCut(in_put=in_put, out_put=out_put, stride=stride, expansion=expansion)
        else:
            self.short_cut = nn.Sequential()

        self.relu = nn.ReLU()

    def forward(self, x):
        in_put = x
        out = self.conv1x1_1(in_put)
        out = self.conv3x3(out)
        out = self.conv1x1_2(out)

        out += self.short_cut(in_put)
        out = self.relu(out)
        return out

class ResUNet(nn.Module):
    def __init__(self, num_classes, expansion=4):
        super().__init__()
        self.exp = expansion
        
        self.enc1 = ResidualBlock(3, 64, expansion=self.exp)
        self.enc2 = ResidualBlock(64*self.exp, 128, expansion=self.exp)
        self.enc3 = ResidualBlock(128*self.exp, 256, expansion=self.exp)
        self.enc4 = ResidualBlock(256*self.exp, 512, expansion=self.exp)
        
        self.pool = nn.MaxPool2d(2)
        
        self.bottom = ResidualBlock(512*self.exp, 1024, expansion=self.exp)
        
        self.up1 = nn.ConvTranspose2d(1024*self.exp, 1024, 2, stride=2)

        self.dec1 = ResidualBlock(1024 + (512*self.exp), 512, expansion=self.exp)
        
        self.up2 = nn.ConvTranspose2d(512*self.exp, 512, 2, stride=2)
        self.dec2 = ResidualBlock(512 + (256*self.exp), 256, expansion=self.exp)
        
        self.up3 = nn.ConvTranspose2d(256*self.exp, 256, 2, stride=2)
        self.dec3 = ResidualBlock(256 + (128*self.exp), 128, expansion=self.exp)
        
        self.up4 = nn.ConvTranspose2d(128*self.exp, 128, 2, stride=2)
        self.dec4 = ResidualBlock(128 + (64*self.exp), 64, expansion=self.exp)
        
        self.out_conv = nn.Conv2d(64*self.exp, num_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x)
        p1 = self.pool(c1)
        c2 = self.enc2(p1)
        p2 = self.pool(c2)
        c3 = self.enc3(p2)
        p3 = self.pool(c3)
        c4 = self.enc4(p3)
        p4 = self.pool(c4)
        
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
    model = ResUNet(num_classes=config['num_class']).to(device)

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