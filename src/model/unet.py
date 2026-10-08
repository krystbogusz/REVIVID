"""Conditional 2D U-Net denoiser of the refiner (inpainting inside holes).

It denoises the residual (gt - coarse) conditioned on the coarse frames, the
hole / validity maps and the backbone features. Conditioning is fed by
channel-concatenation at the input AND re-injected after every downsampler
(zero-initialised 1x1 projections); the timestep enters every residual block
through FiLM.

Frames of a clip arrive as consecutive batch items; ``num_frames`` groups them,
and on ``temporal_levels`` (and in the bottleneck) every residual block is
followed by attention ACROSS the frames, so a window is generated jointly
instead of frame by frame.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .blocks import (
    AttnBlock,
    Downsample,
    Normalize,
    TemporalAttention,
    TimeConditionedResBlock,
    TimestepEmbedding,
    Upsample,
)


class UNetBlock(nn.Module):
    """Timestep-conditioned residual block, optionally followed by attention
    across the frames of the clip."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, temporal: bool):
        super().__init__()
        self.res = TimeConditionedResBlock(in_ch, out_ch, time_dim)
        self.temporal = TemporalAttention(out_ch) if temporal else None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, num_frames: int) -> torch.Tensor:
        x = self.res(x, t_emb)
        if self.temporal is not None:
            x = self.temporal(x, num_frames)
        return x


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        out_channels: int,
        base_channels: int = 48,
        channel_mult: Sequence[int] = (1, 2, 3),
        num_res_blocks: int = 2,
        temporal_levels: Sequence[int] = (1, 2),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.size_factor = 2 ** (len(channel_mult) - 1)
        time_dim = base_channels * 4
        self.time_embed = TimestepEmbedding(base_channels, time_dim)
        self.conv_in = nn.Conv2d(in_channels + cond_channels, base_channels, 3, 1, 1)

        # Conditioning re-injected after the downsampler leaving each level
        # (zero-initialised, so it starts as a no-op).
        self.cond_projs = nn.ModuleList()
        for mult in channel_mult[:-1]:
            proj = nn.Conv2d(cond_channels, base_channels * mult, 1)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.cond_projs.append(proj)

        self.down_blocks = nn.ModuleList()
        self.down_samplers = nn.ModuleList()
        chans = [base_channels]
        cur = base_channels
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(UNetBlock(cur, out_ch, time_dim, level in temporal_levels))
                cur = out_ch
                chans.append(cur)
            self.down_blocks.append(blocks)
            last = level == len(channel_mult) - 1
            self.down_samplers.append(None if last else Downsample(cur))
            if not last:
                chans.append(cur)

        self.mid1 = UNetBlock(cur, cur, time_dim, temporal=bool(temporal_levels))
        self.mid_attn = AttnBlock(cur)
        self.mid2 = UNetBlock(cur, cur, time_dim, temporal=False)

        self.up_blocks = nn.ModuleList()
        self.up_samplers = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(UNetBlock(cur + chans.pop(), out_ch, time_dim, level in temporal_levels))
                cur = out_ch
            self.up_blocks.append(blocks)
            self.up_samplers.append(Upsample(cur) if level != 0 else None)

        self.out_norm = Normalize(cur)
        self.conv_out = nn.Conv2d(cur, out_channels, 3, 1, 1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    @staticmethod
    def _run(block: UNetBlock, h, t_emb, num_frames):
        # Gradient checkpointing: the refiner runs on full HR frames.
        return checkpoint(block, h, t_emb, num_frames, use_reentrant=False)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, num_frames: int = 1
    ) -> torch.Tensor:
        """``x`` (N*T, C, H, W) with the T frames of each clip consecutive."""
        t_emb = self.time_embed(t)
        x = torch.cat([x, cond], dim=1)

        h0, w0 = x.shape[-2:]
        f = self.size_factor
        pad_h, pad_w = (f - h0 % f) % f, (f - w0 % f) % f
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        cond_p = x[:, self.in_channels :]

        h = self.conv_in(x)
        skips = [h]
        for level, (blocks, sampler) in enumerate(zip(self.down_blocks, self.down_samplers)):
            for block in blocks:
                h = self._run(block, h, t_emb, num_frames)
                skips.append(h)
            if sampler is not None:
                h = sampler(h)
                c = F.interpolate(cond_p, size=h.shape[-2:], mode="bilinear", align_corners=False)
                h = h + self.cond_projs[level](c)
                skips.append(h)

        h = self._run(self.mid1, h, t_emb, num_frames)
        h = self.mid_attn(h)
        h = self._run(self.mid2, h, t_emb, num_frames)

        for blocks, sampler in zip(self.up_blocks, self.up_samplers):
            for block in blocks:
                h = self._run(block, torch.cat([h, skips.pop()], dim=1), t_emb, num_frames)
            if sampler is not None:
                h = sampler(h)

        out = self.conv_out(F.silu(self.out_norm(h)))
        return out[..., :h0, :w0]
