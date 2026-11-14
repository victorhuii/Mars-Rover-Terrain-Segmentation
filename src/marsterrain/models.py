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


class MobileNetUNet(nn.Module):
    def __init__(self, pretrained=False, classes=4):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = mobilenet_v3_small(weights=weights).features[:12]
        self.dec4 = ConvBlock(96 + 48, 64)
        self.dec3 = ConvBlock(64 + 24, 48)
        self.dec2 = ConvBlock(48 + 16, 32)
        self.dec1 = ConvBlock(32 + 16, 24)
        self.output = nn.Conv2d(24, classes, 1)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        x = (image - self.mean) / self.std
        features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in (0, 1, 3, 8, 11): features.append(x)
        a, b, c, d, x = features
        x = self.dec4(torch.cat((F.interpolate(x, size=d.shape[-2:], mode='bilinear', align_corners=False), d), dim=1))
        x = self.dec3(torch.cat((F.interpolate(x, size=c.shape[-2:], mode='bilinear', align_corners=False), c), dim=1))
        x = self.dec2(torch.cat((F.interpolate(x, size=b.shape[-2:], mode='bilinear', align_corners=False), b), dim=1))
        x = self.dec1(torch.cat((F.interpolate(x, size=a.shape[-2:], mode='bilinear', align_corners=False), a), dim=1))
        return self.output(F.interpolate(x, size=image.shape[-2:], mode='bilinear', align_corners=False))


def make_model(architecture, channels=16, pretrained=False):
    if architecture == 'unet': return UNet(channels=channels)
    if architecture == 'mobilenet_v3_small': return MobileNetUNet(pretrained=pretrained)
    raise ValueError(f'Unknown architecture: {architecture}')
