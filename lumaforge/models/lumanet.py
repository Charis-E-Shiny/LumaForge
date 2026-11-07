"""LumaNet: a compact U-Net built from depthwise-separable residual blocks.

Hybrid mode:
    The classical render is the base layer. The network sees the packed RAW, the render and
    a noise map, then predicts (1) a global 3x4 colour transform and (2) a local residual.
    Colour and tone stay anchored to a physically meaningful render, so the network spends
    its capacity on restoration rather than on relearning the whole camera pipeline.

Learned mode:
    The same backbone maps packed RAW directly to sRGB with no classical input. This is the
    "learned ISP" arm of the comparison.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class NetConfig:
    width: int = 32
    hybrid: bool = True
    use_noise_map: bool = True
    use_color_head: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class ResidualBlock(nn.Module):
    """Inverted-bottleneck-free separable block: cheap on CPUs and easy to quantize."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pointwise(self.activation(self.depthwise(x)))


class Stage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int = 2) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.activation(self.project(x)))


class LumaNet(nn.Module):
    """Inputs are at half resolution (packed RAW); output is at full resolution."""

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config
        w = config.width
        in_channels = 4
        if config.hybrid:
            in_channels += 12  # base render after pixel-unshuffle
        if config.use_noise_map:
            in_channels += 1
        self.enc1 = Stage(in_channels, w)
        self.down1 = nn.Conv2d(w, w * 2, 2, stride=2)
        self.enc2 = Stage(w * 2, w * 2)
        self.down2 = nn.Conv2d(w * 2, w * 4, 2, stride=2)
        self.enc3 = Stage(w * 4, w * 4)
        self.down3 = nn.Conv2d(w * 4, w * 4, 2, stride=2)
        self.bottleneck = Stage(w * 4, w * 4, blocks=3)
        self.dec3 = Stage(w * 8, w * 4)
        self.dec2 = Stage(w * 6, w * 2)
        self.dec1 = Stage(w * 3, w)
        self.head = nn.Conv2d(w, 12, 3, padding=1)  # 12 channels = 3 RGB at 2x via shuffle
        self.color_head = None
        if config.hybrid and config.use_color_head:
            self.color_head = nn.Sequential(nn.Linear(w * 4, 32), nn.ReLU(inplace=True), nn.Linear(32, 12))
            nn.init.zeros_(self.color_head[-1].weight)
            nn.init.zeros_(self.color_head[-1].bias)
        if config.hybrid:
            # Start from "output = base" so training refines rather than relearns.
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    @staticmethod
    def _up(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

    def forward(self, packed: torch.Tensor, noise: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        parts = [packed]
        if self.config.hybrid:
            parts.append(F.pixel_unshuffle(base, 2))
        if self.config.use_noise_map:
            parts.append(noise)
        e1 = self.enc1(torch.cat(parts, dim=1))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.dec3(torch.cat([self._up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2), e1], dim=1))
        residual = F.pixel_shuffle(self.head(d1), 2)
        if not self.config.hybrid:
            return residual
        out = base
        if self.color_head is not None:
            params = self.color_head(b.mean(dim=(2, 3)))
            matrix = params[:, :9].view(-1, 3, 3)
            offset = params[:, 9:].view(-1, 3, 1, 1)
            # (I + delta) * rgb + offset, written so ONNX keeps dynamic spatial axes.
            height, width = base.shape[2], base.shape[3]
            flat = base.flatten(2)
            adjusted = flat + torch.bmm(matrix, flat)
            out = adjusted.view(-1, 3, height, width) + offset
        return out + residual


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
