"""Neural-network building blocks shared by the backbone and the refiner UNet."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Standard transformer sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(sinusoidal_embedding(t, self.dim))


def Normalize(channels: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels, eps=1e-4)


class ResidualBlockNoBN(nn.Module):
    """Residual block without batch-norm (BasicVSR style)."""

    def __init__(self, num_feat: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


def make_layer(block, num_blocks: int, **kwargs) -> nn.Sequential:
    return nn.Sequential(*[block(**kwargs) for _ in range(num_blocks)])


class TimeConditionedResBlock(nn.Module):
    """ResBlock with FiLM-style timestep conditioning (used inside the UNet)."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = Normalize(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = Normalize(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Single-head spatial self-attention (the UNet bottleneck)."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = Normalize(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        # (b, 1 head, h*w, c), contiguous: only this layout lets PyTorch pick the
        # memory-efficient kernel, which never materialises the (h*w) x (h*w)
        # attention matrix (tens of GB at full HD).
        q, k, v = (t.reshape(b, 1, c, h * w).transpose(2, 3).contiguous() for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(2, 3).reshape(b, c, h, w)
        return x + self.proj(out)


class TemporalAttention(nn.Module):
    """Self-attention ACROSS FRAMES at every pixel (used inside the UNet).

    The UNet sees a clip as a batch of N*T frames; this block lets every pixel
    attend to the same location in the other frames of its clip, so content the
    refiner invents is generated for the whole window jointly instead of frame
    by frame (which flickers). No temporal position encoding: the block is
    permutation-invariant over frames, so any window length works.
    Zero-initialised output: the block starts as identity.
    """

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.heads = heads if channels % heads == 0 else 1
        self.norm = Normalize(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        if num_frames <= 1:
            return x
        bt, c, h, w = x.shape
        t, b, nh = num_frames, bt // num_frames, self.heads
        d = c // nh
        qkv = self.qkv(self.norm(x)).view(b, t, 3, nh, d, h, w)
        # -> (3, b*h*w, heads, t, d): one length-t sequence per pixel
        qkv = qkv.permute(2, 0, 5, 6, 3, 1, 4).reshape(3, b * h * w, nh, t, d)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        out = out.view(b, h, w, nh, t, d).permute(0, 4, 3, 5, 1, 2).reshape(bt, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))
