import sys
import os
import yaml
import torch.nn.functional as F
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.append(root_path)
from F_KeypointDetection.Util.Util import *
from F_KeypointDetection.Util.Draw_Graph import *

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, input, output, stride=1, downsampling=None):
        super(BasicBlock, self).__init__()
        self.conv3x3_1 = nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(output, momentum=0.1)
        self.conv3x3_2 = nn.Conv2d(output, output, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(output, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.downsampling = downsampling
    def forward(self, x):
        residual = x
        out = self.relu(self.bn_1(self.conv3x3_1(x)))
        out = self.bn_2(self.conv3x3_2(out))
        if self.downsampling is not None: residual = self.downsampling(x)
        out += residual
        return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, input, output, stride=1, downsampling=None):
        super(Bottleneck, self).__init__()
        self.conv1x1_1 = nn.Conv2d(input, output, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output, momentum=0.1)
        self.conv3x3_1 = nn.Conv2d(output, output, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output, momentum=0.1)
        self.conv1x1_2 = nn.Conv2d(output, output * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output * self.expansion, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.downsampling = downsampling
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1x1_1(x)))
        out = self.relu(self.bn2(self.conv3x3_1(out)))
        out = self.bn3(self.conv1x1_2(out))
        if self.downsampling is not None: residual = self.downsampling(x)
        out += residual
        return self.relu(out)

class HRModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_in_channels, num_channels, fuse_method, multi_scale_output=True):
        super(HRModule, self).__init__()
        self.num_in_channels, self.fuse_method, self.num_branches = num_in_channels, fuse_method, num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsampling = None
        if stride != 1 or self.num_in_channels[branch_index] != num_channels[branch_index] * block.expansion:
            downsampling = nn.Sequential(
                nn.Conv2d(self.num_in_channels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=0.1),
            )
        layers = [block(self.num_in_channels[branch_index], num_channels[branch_index], stride, downsampling)]
        self.num_in_channels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_in_channels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branch, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branch):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1: return None
        fuse_layers = []
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.num_in_channels[j], self.num_in_channels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(self.num_in_channels[i], momentum=0.1),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i: fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(self.num_in_channels[j], self.num_in_channels[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(self.num_in_channels[i], momentum=0.1)))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(self.num_in_channels[j], self.num_in_channels[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(self.num_in_channels[j], momentum=0.1),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self): return self.num_in_channels

    def forward(self, x):
        if self.num_branches == 1: return [self.branches[0](x[0])]
        for i in range(self.num_branches): x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                y = y + x[j] if i == j else y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse

class HighResolutionNet(nn.Module):
    def __init__(self, cfg_name='W48'):
        super(HighResolutionNet, self).__init__()
        configs = {
            'W18': [[1, 1, 'BOTTLENECK', [4], [18], 'SUM'], [1, 2, 'BASIC', [4, 4], [18, 36], 'SUM'], [4, 3, 'BASIC', [4, 4, 4], [18, 36, 72], 'SUM'], [3, 4, 'BASIC', [4, 4, 4, 4], [18, 36, 72, 144], 'SUM']],
            'W48': [[1, 1, 'BOTTLENECK', [4], [64], 'SUM'], [1, 2, 'BASIC', [4, 4], [48, 96], 'SUM'], [4, 3, 'BASIC', [4, 4, 4], [48, 96, 192], 'SUM'], [3, 4, 'BASIC', [4, 4, 4, 4], [48, 96, 192, 384], 'SUM']]
        }
        self.stage_config = configs[cfg_name]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)
        
        # Stage 2
        self.transition1 = self._make_transition_layer([256], [48, 96])
        self.stage2, pre_channels = self._make_stage(self.stage_config[1], [48, 96])

        # Stage 3
        self.transition2 = self._make_transition_layer(pre_channels, [48, 96, 192])
        self.stage3, pre_channels = self._make_stage(self.stage_config[2], [48, 96, 192])

        # Stage 4
        self.transition3 = self._make_transition_layer(pre_channels, [48, 96, 192, 384])
        self.stage4, pre_channels = self._make_stage(self.stage_config[3], [48, 96, 192, 384])

    def _make_transition_layer(self, pre_ch, cur_ch):
        layers = []
        for i in range(len(cur_ch)):
            if i < len(pre_ch):
                if cur_ch[i] != pre_ch[i]:
                    layers.append(nn.Sequential(nn.Conv2d(pre_ch[i], cur_ch[i], 3, 1, 1, bias=False), nn.BatchNorm2d(cur_ch[i]), nn.ReLU(inplace=True)))
                else: layers.append(None)
            else:
                conv3x3s = [nn.Sequential(nn.Conv2d(pre_ch[-1], cur_ch[i], 3, 2, 1, bias=False), nn.BatchNorm2d(cur_ch[i]), nn.ReLU(inplace=True))]
                layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, 1, stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks): layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, cfg, in_ch):
        modules = []
        for i in range(cfg[0]):
            modules.append(HRModule(cfg[1], BasicBlock if cfg[2]=='BASIC' else Bottleneck, cfg[3], in_ch, cfg[4], cfg[5]))
            in_ch = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), in_ch

class HRNet_Keypoint(HighResolutionNet):
    def __init__(self, cfg_name='W48', num_keypoints=17):
        super(HRNet_Keypoint, self).__init__(cfg_name=cfg_name)
        self.last_channels = 48
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(self.last_channels, self.last_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.last_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.last_channels, num_keypoints, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = [self.transition1[i](x) if self.transition1[i] else x for i in range(2)]
        y_list = self.stage2(x_list)
        x_list = [self.transition2[i](y_list[-1]) if self.transition2[i] else y_list[i] for i in range(3)]
        y_list = self.stage3(x_list)
        x_list = [self.transition3[i](y_list[-1]) if self.transition3[i] else y_list[i] for i in range(4)]
        y_list = self.stage4(x_list)

        out = self.keypoint_head(y_list[0])
        return out
    

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, '..', 'Util', 'config.yaml')
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HRNet_Keypoint().to(device)

    train_loader, val_loader, test_loader = keypointdetection_data_loader(
        config['load_path'], config['batch_size']
    )

    graph = Draw_Graph(model=model, save_path=config['save_path'], patience=config['patience'])
    #model.load_state_dict(torch.load(os.path.join(config['save_path'], "Best_Accuracy_Validation.pth")))
    train_model(
        device=device, model=model, train_loader=train_loader, 
        val_loader=val_loader, graph=graph, epochs=config['epoch'], 
        lr=1e-4, patience=config['patience'], save_path=config['save_path']
    )

    test_and_visualize_all(model, test_loader, device, model_path=os.path.join(config['save_path'], "Best_Accuracy_Validation.pth"))

if __name__ == "__main__":
    main()