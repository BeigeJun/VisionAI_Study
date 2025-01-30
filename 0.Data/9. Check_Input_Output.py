import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_put, reduce=4):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_put, max(1, int(in_put//reduce))),
            nn.SiLU(),
            nn.Linear(max(1, int(in_put//reduce)), in_put),
            nn.Sigmoid()
        )

    def forward(self, x):
        se = self.squeeze(x)
        se = torch.flatten(se, 1)
        se = self.excitation(se)
        se = se.unsqueeze(dim=2).unsqueeze(dim=3)
        out = se * x
        return out


in_channels = 16
se_block = SEBlock(in_put=in_channels)
input_tensor = torch.randn(2, 16, 16, 16)
output_tensor = se_block(input_tensor)
print("입력 형태:", input_tensor.shape)
print("출력 형태:", output_tensor.shape)
