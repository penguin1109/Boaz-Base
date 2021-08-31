import torch.nn as nn
import torch
import torch.nn.functional as F

from squeeze_excitation import Attention
class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride = 1, use_batchnorm = True):
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(bn)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.attention1 = Attention(in_channels = in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.attention2 = Attention(in_channels = out_channels)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = 2, mode = 'nearest')

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attentopm
