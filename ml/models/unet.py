"""U-Net baseline for image restoration (denoising / super-resolution-at-fixed-size).
Chosen as the primary baseline per project spec: fast to train even on CPU,
strong at preserving edges/structure when trained with an L1 loss, which
matters for defect-preserving restoration in semiconductor inspection.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Configurable-depth U-Net. `base_channels` and `depth` come from
    configs/config.yaml (models.unet.*) — not hardcoded at call sites."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth

        chs = [base_channels * (2 ** i) for i in range(depth)]

        self.downs = nn.ModuleList()
        prev_ch = in_channels
        for ch in chs:
            self.downs.append(ConvBlock(prev_ch, ch))
            prev_ch = ch
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(chs[-1], chs[-1] * 2)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        rev_chs = list(reversed(chs))
        prev_ch = chs[-1] * 2
        for ch in rev_chs:
            self.ups.append(nn.ConvTranspose2d(prev_ch, ch, kernel_size=2, stride=2))
            self.up_convs.append(ConvBlock(ch * 2, ch))
            prev_ch = ch

        self.out_conv = nn.Conv2d(chs[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        h = x
        for down in self.downs:
            h = down(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for up, up_conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            h = up(h)
            # Handle odd input sizes via center-crop/pad to skip's spatial size
            if h.shape[-2:] != skip.shape[-2:]:
                h = nn.functional.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = up_conv(h)

        out = self.out_conv(h)
        # Residual learning: predict the clean-degraded residual, add back input.
        # This helps preserve structure (defects/edges) rather than hallucinating
        # a wholly new image — important for inspection use cases.
        return torch.clamp(x + out, 0.0, 1.0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
