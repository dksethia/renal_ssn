import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_prob=0.2):
        super().__init__()

        self.inc = DoubleConv(in_channels, 32, dropout_prob=dropout_prob)
        self.down1 = Down(32, 64, dropout_prob=dropout_prob)
        self.down2 = Down(64, 128, dropout_prob=dropout_prob)
        self.down3 = Down(128, 256, dropout_prob=dropout_prob)
        self.down4 = Down(256, 512, dropout_prob=dropout_prob)
        self.down5 = Down(512, 1024, dropout_prob=dropout_prob)
        self.up1 = Up(1024, 512, dropout_prob=dropout_prob)
        self.up2 = Up(512, 256, dropout_prob=dropout_prob)
        self.up3 = Up(256, 128, dropout_prob=dropout_prob)
        self.up4 = Up(128, 64, dropout_prob=dropout_prob)
        self.up5 = Up(64, 32, dropout_prob=dropout_prob)
        self.outc = OutConv(32, num_classes)

    def forward(self, x):
        d1 = self.inc(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)

        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        logits = self.outc(u5)

        return logits