# Code implemented based on https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """ (Conv => InstanceNorm => LeakyReLU) * 2 """

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_prob=0.2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.Dropout(p=dropout_prob),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.Dropout(p=dropout_prob),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """ MaxPool => DoubleConv """
    
    def __init__(self, in_channels, out_channels, dropout_prob=0.2):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_prob=dropout_prob)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """ TransposeConv => Concat => DoubleConv """

    def __init__(self, in_channels, out_channels, dropout_prob=0.2):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.conv = DoubleConv(in_channels + out_channels, out_channels, dropout_prob=dropout_prob)

    def forward(self, x1, x2):
        x_up = self.dropout(self.up(x1))
        x = torch.cat([x2, x_up], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """ 1x1 Conv """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)        