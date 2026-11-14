"""Compact U-Net segmentation models."""
import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, channels=16, classes=4):
        super().__init__()
        self.enc1 = ConvBlock(3, channels)
        self.enc2 = ConvBlock(channels, channels * 2)
        self.enc3 = ConvBlock(channels * 2, channels * 4)
        self.middle = ConvBlock(channels * 4, channels * 8)
        self.dec3 = ConvBlock(channels * 12, channels * 4)
        self.dec2 = ConvBlock(channels * 6, channels * 2)
        self.dec1 = ConvBlock(channels * 3, channels)
        self.output = nn.Conv2d(channels, classes, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.enc3(F.max_pool2d(b, 2))
        d = self.middle(F.max_pool2d(c, 2))
        d = self.dec3(torch.cat((F.interpolate(d, size=c.shape[-2:], mode='bilinear', align_corners=False), c), dim=1))
        d = self.dec2(torch.cat((F.interpolate(d, size=b.shape[-2:], mode='bilinear', align_corners=False), b), dim=1))
        d = self.dec1(torch.cat((F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a), dim=1))
        return self.output(d)
