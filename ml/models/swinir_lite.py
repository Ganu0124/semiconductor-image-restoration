"""SwinIR-Lite: a real, working, windowed-self-attention restoration network,
architecturally in the SwinIR/Restormer family (shifted-window multi-head
self-attention + residual conv blocks), but sized down (`embed_dim`,
`depths` in configs/config.yaml -> models.swinir) so it can actually train on
a CPU in DEV_MODE within this environment.

For a full-size SwinIR/Restormer trained to paper-scale, run this same
training pipeline (ml/training/train.py --model swinir) on a CUDA GPU with
larger embed_dim/depths — the model registry and training/eval code do not
change, only the config. See docs/training.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int = 4):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]

        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C).permute(0, 1, 3, 2, 4, 5)
        windows = x.reshape(-1, ws * ws, C)  # (num_windows*B, ws*ws, C)

        qkv = self.qkv(windows).reshape(windows.shape[0], ws * ws, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(windows.shape[0], ws * ws, C)
        out = self.proj(out)

        out = out.view(B, Hp // ws, Wp // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(B, Hp, Wp, C)
        if pad_h or pad_w:
            out = out[:, :H, :W, :]
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        # x: (B, H, W, C)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SwinIRLite(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 embed_dim: int = 60, window_size: int = 8, depths: list[int] | None = None):
        super().__init__()
        depths = depths or [4, 4, 4, 4]
        self.embed = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.blocks = nn.ModuleList([
            SwinBlock(embed_dim, window_size) for _ in range(sum(depths))
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.reconstruct = nn.Conv2d(embed_dim, out_channels, 3, padding=1)

    def forward(self, x):
        feat = self.embed(x)  # (B, C, H, W)
        feat = feat.permute(0, 2, 3, 1)  # (B, H, W, C)
        for blk in self.blocks:
            feat = blk(feat)
        feat = self.norm(feat)
        feat = feat.permute(0, 3, 1, 2)  # (B, C, H, W)
        out = self.reconstruct(feat)
        return torch.clamp(x + out, 0.0, 1.0)  # residual restoration


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
