import torch
import torch.nn as nn

class SCSEModule(nn.Module):
    def __init__(self, in_channels, mode, reduction=16):
        super().__init__()
        self.mode = mode
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        if self.mode == 'cat':
            return nn.cat((x * self.cSE(x), x * self.sSE(x)), dim = 1)
        elif self.mode == 'max':
            return torch.max((x*self.cSE, x*self.sSE), dim = 1)

class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = SCSEModule(in_channels = in_channels)
    
    def forward(self, x):
        return self.attention(x)