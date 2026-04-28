import sys
import os
import yaml
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.append(root_path)
from E_Segmentation.Util.Util import *
from E_Segmentation.Util.Draw_Graph import *
from torchvision import models

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.msa = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        x = x + self.msa(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class TransUNet(nn.Module):
    def __init__(self, num_classes=1, img_size=512):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        self.early_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) 
        self.pool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.patch_size = img_size // 16
        num_patches = self.patch_size ** 2
        
        self.bottleneck_conv = nn.Conv2d(1024, 512, kernel_size=1)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, 512))
        self.transformer_layers = nn.Sequential(
            *[TransformerBlock(512, 8, 1024) for _ in range(4)]
        )

        self.dec1 = DecoderBlock(512 + 512, 256)
        self.dec2 = DecoderBlock(256 + 256, 128)
        self.dec3 = DecoderBlock(128 + 64, 64)
        
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
            s0 = x 
            s1 = self.early_conv(x)
            
            x_p = self.pool(s1)
            s2 = self.layer1(x_p)
            s3 = self.layer2(s2)
            s4 = self.layer3(s3)

            b = self.bottleneck_conv(s4) 
            grid_size = b.shape[-1]
            b = b.flatten(2).transpose(1, 2)
            b = b + self.pos_embedding
            b = self.transformer_layers(b)
            b = b.transpose(1, 2).reshape(-1, 512, grid_size, grid_size)

            d1 = self.dec1(b, s3)
            d2 = self.dec2(d1, s2)
            d3 = self.dec3(d2, s1)

            output = self.final_upsample(d3)
            output = self.final_conv(output)
            
            return output
    


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, '..', 'Util', 'config.yaml')
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransUNet(num_classes=config['num_class'], img_size=config['input_size']).to(device)

    train_loader, val_loader, test_loader = segmentation_data_loader(
        config['load_path'], config['batch_size'], config['input_size']
    )

    #graph = Draw_Graph(model=model, save_path=config['save_path'], patience=config['patience'])
    model.load_state_dict(torch.load(os.path.join(config['save_path'], "Best_Accuracy_Validation.pth")))
    # train_model(
    #     device=device, model=model, train_loader=train_loader, 
    #     val_loader=val_loader, graph=graph, epochs=config['epoch'], 
    #     lr=1e-4, patience=config['patience'], save_path=config['save_path']
    # )

    #visualize_random_10(model, device, config['load_path'], config['input_size'])
    visualize_all_test_set(model, device, config['load_path'], config['input_size'])
if __name__ == "__main__":
    main()