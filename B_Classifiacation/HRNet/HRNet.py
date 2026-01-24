import torch
import torch.nn as nn
import torch.nn.functional as F

config = {
    'GPUS': (0, 1, 2, 3),
    'LOG_DIR': 'log/',
    'DATA_DIR': '',
    'OUTPUT_DIR': 'output/',
    'WORKERS': 4,
    'PRINT_FREQ': 1000,

    'MODEL': {
        'NAME': 'cls_hrnet',
        'IMAGE_SIZE': [224, 224],
        'EXTRA': {
            'WITH_HEAD': True,
            'STAGE1': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 1,
                'BLOCK': 'BOTTLENECK',
                'NUM_BLOCKS': [1],
                'NUM_CHANNELS': [32],
                'FUSE_METHOD': 'SUM'
            },
            'STAGE2': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 2,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [2, 2],
                'NUM_CHANNELS': [16, 32],
                'FUSE_METHOD': 'SUM'
            },
            'STAGE3': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 3,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [2, 2, 2],
                'NUM_CHANNELS': [16, 32, 64],
                'FUSE_METHOD': 'SUM'
            },
            'STAGE4': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 4,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [2, 2, 2, 2],
                'NUM_CHANNELS': [16, 32, 64, 128],
                'FUSE_METHOD': 'SUM'
            }
        }
    },

    'CUDNN': {
        'BENCHMARK': True,
        'DETERMINISTIC': False,
        'ENABLED': True
    },

    'DATASET': {
        'DATASET': 'imagenet',
        'DATA_FORMAT': 'jpg',
        'ROOT': 'data/imagenet/',
        'TEST_SET': 'val',
        'TRAIN_SET': 'train'
    },

    'TEST': {
        'BATCH_SIZE_PER_GPU': 32,
        'MODEL_FILE': ''
    },

    'TRAIN': {
        'BATCH_SIZE_PER_GPU': 32,
        'BEGIN_EPOCH': 0,
        'END_EPOCH': 100,
        'RESUME': True,
        'LR_FACTOR': 0.1,
        'LR_STEP': [30, 60, 90],
        'OPTIMIZER': 'sgd',
        'LR': 0.05,
        'WD': 0.0001,
        'MOMENTUM': 0.9,
        'NESTEROV': True,
        'SHUFFLE': True
    },

    'DEBUG': {
        'DEBUG': False
    }
}

class BasicBlock(nn.Module):
    def __init__(self, input, output, stride=1, downsampling=None):
        super(BasicBlock, self).__init__()
        self.expansion = 1

        self.conv3x3_1 = nn.Conv2d(input, output, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(output, momentum=0.1)

        self.conv3x3_2 = nn.Conv2d(output, output, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(output, momentum=0.1)

        self.relu = nn.ReLU(inplace=True)
        self.downsampling = downsampling
        self.stride = stride

    def forward(self, x):
        residual = x
        output = self.conv3x3_1(x)
        output = self.bn_1(output)
        output = self.relu(output)

        output = self.conv3x3_2(output)
        output = self.bn_2(output)

        if self.downsampling is not None:
            residual = self.downsampling(x)

        output += residual
        output = self.relu(output)
        return output

class Bottleneck(nn.Module):
    def __init__(self, input, output, stride=1, downsampling=None):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.conv1x1_1 = nn.Conv2d(input, output, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output, momentum=0.1)

        self.conv3x3_1 = nn.Conv2d(output, output, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output, momentum=0.1)

        self.conv1x1_2 = nn.Conv2d(output, output * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output * self.expansion, momentum=0.1)

        self.relu = nn.ReLU(inplace=True)
        self.downsampling = downsampling
        self.stride = stride

    def forward(self, x):
        residual = x
        output = self.conv1x1_1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv3x3_1(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv1x1_2(output)
        output = self.bn3(output)

        if self.downsampling is not None:
            residual = self.downsampling(x)

        output += residual
        output = self.relu(output)
        return output




class HRModule(nn.Module):
    def __int__(self, num_branches, blocks, num_blocks, num_in_channels, num_channels, fuse_method, multi_scale_output=True):
        super(HRModule, self).__int__()

        self.num_in_channels = num_in_channels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    # 'STAGE1': {
    #     'NUM_MODULES': 1,
    #     'NUM_BRANCHES': 1,
    #     'BLOCK': 'BOTTLENECK',
    #     'NUM_BLOCKS': [1],
    #     'NUM_CHANNELS': [32],
    #     'FUSE_METHOD': 'SUM'
    # },

    def _make_one_brach(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsampling = None
        if stride != 1 or self.num_in_channels[branch_index] != num_channels[branch_index * block.expansion]:
            downsampling = nn.Sequential(
                nn.Conv2d(self.num_in_channels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=0.1),
            )

        layers = []
        layers.append(block(self.num_in_channels[branch_index], num_channels[branch_index], stride, downsampling))
        self.num_in_channels[branch_index] = num_channels * block.expansion

        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_in_channels[branch_index], num_channels[branch_index]))

        return  nn.Sequential(*layers)

    def _make_branches(self, num_branch, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branch):
            branches.append(self._make_one_brach(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches) #이걸 쓰는이유는 Sequential은 append로만 이루어진 간단한 구조인데 이 branch는 아니기 때문 즉 어려운 구조일때 이걸 쓴다?


    def _make_fuse_layers(self):
        if self.num_branches == 1: #단일 브랜치는 Fusion 불필요
            return 0

        num_branches = self.num_branches
        num_in_channels = self.num_in_channels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []











