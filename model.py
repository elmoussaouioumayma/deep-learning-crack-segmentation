"""A compact U-Net for crack segmentation (about 1.9 million parameters)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Encoder-decoder with skip connections; output = one logit per pixel (crack / no crack)."""

    def __init__(self, base=16, depth=4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([ConvBlock(3, chs[0])] + [ConvBlock(chs[i], chs[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([ConvBlock(2 * chs[i], chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips[:-1])):
            x = up(x)
            x = dec(torch.cat([x, skip], 1))
        return self.head(x)


def dice_bce_loss(logits, target, pos_weight=5.0):
    """Weighted BCE (cracks are only ~3 % of the pixels) + soft Dice (overlap of thin structures)."""
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(pos_weight, device=logits.device))
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
