import sys
import os
import yaml
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.append(root_path)
from E_Segmentation.Util.Util import *
from E_Segmentation.Util.Draw_Graph import *

class VGGBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        nb_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.L0_0 = VGGBlock(3, nb_filter[0])
        self.L1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.L2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.L3_0 = VGGBlock(nb_filter[2], nb_filter[3])
        self.L4_0 = VGGBlock(nb_filter[3], nb_filter[4])

        self.L0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.L1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.L2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.L3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.L0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.L1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.L2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.L0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.L1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.L0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.L0_0(x)
        x1_0 = self.L1_0(self.pool(x0_0))
        x0_1 = self.L0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.L2_0(self.pool(x1_0))
        x1_1 = self.L1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.L0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.L3_0(self.pool(x2_0))
        x2_1 = self.L2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.L1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.L0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.L4_0(self.pool(x3_0))
        x3_1 = self.L3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.L2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.L1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.L0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return [out1, out2, out3, out4]
        
        return self.final(x0_4)
    

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, '..', 'Util', 'config.yaml')
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlus(num_classes=config['num_class']).to(device)

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