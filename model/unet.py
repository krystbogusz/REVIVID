"""Conditional 2D U-Net denoiser.

A single unified v-prediction denoiser used for all three tasks (restoration,
VFI, inpainting). It denoises a high-frequency residual conditioned on the
coarse backbone output (coarse frame, hole mask, backbone features, and the
temporal frame-mask embedding).

Conditioning is fed by channel-concatenation at the input; the timestep is
injected via FiLM inside every residual block.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .blocks import (
    AttnBlock,
    Downsample,
    Normalize,
    TimeConditionedResBlock,
    TimestepEmbedding,
    Upsample,
)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        cond_channels: int = 0,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_levels: Sequence[int] = (2,),
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.use_checkpoint = use_checkpoint
        self.size_factor = 2 ** (len(channel_mult) - 1)

        time_dim = base_channels * 4
        self.time_embed = TimestepEmbedding(base_channels, time_dim)

        self.conv_in = nn.Conv2d(in_channels + cond_channels, base_channels, 3, 1, 1)

        # ----- Encoder -----
        self.down_blocks = nn.ModuleList()
        self.down_samplers = nn.ModuleList()
        chans = [base_channels]
        cur = base_channels
        num_levels = len(channel_mult)
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(TimeConditionedResBlock(cur, out_ch, time_dim, dropout))
                cur = out_ch
                if level in attn_levels:
                    blocks.append(AttnBlock(cur))
                chans.append(cur)
            self.down_blocks.append(blocks)
            if level != num_levels - 1:
                self.down_samplers.append(Downsample(cur))
                chans.append(cur)
            else:
                self.down_samplers.append(None)

        # ----- Bottleneck -----
        self.mid_block1 = TimeConditionedResBlock(cur, cur, time_dim, dropout)
        self.mid_attn = AttnBlock(cur)
        self.mid_block2 = TimeConditionedResBlock(cur, cur, time_dim, dropout)

        # ----- Decoder -----
        self.up_blocks = nn.ModuleList()
        self.up_samplers = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(TimeConditionedResBlock(cur + chans.pop(), out_ch, time_dim, dropout))
                cur = out_ch
                if level in attn_levels:
                    blocks.append(AttnBlock(cur))
            self.up_blocks.append(blocks)
            if level != 0:
                self.up_samplers.append(Upsample(cur))
            else:
                self.up_samplers.append(None)

        self.out_norm = Normalize(cur)
        self.conv_out = nn.Conv2d(cur, out_channels, 3, 1, 1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        t_emb = self.time_embed(t)
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        # Pad to a multiple of the down-sampling factor so any H/W works, then
        # crop the result back to the original spatial size.
        h0, w0 = x.shape[-2:]
        f = self.size_factor
        pad_h = (f - h0 % f) % f
        pad_w = (f - w0 % f) % f
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        h = self.conv_in(x)

        skips = [h]
        for blocks, sampler in zip(self.down_blocks, self.down_samplers):
            for block in blocks:
                if isinstance(block, TimeConditionedResBlock):
                    if self.use_checkpoint:
                        h = grad_checkpoint(block, h, t_emb, use_reentrant=False)
                    else:
                        h = block(h, t_emb)
                    skips.append(h)
                else:
                    h = block(h)
            if sampler is not None:
                h = sampler(h)
                skips.append(h)

        if self.use_checkpoint:
            h = grad_checkpoint(self.mid_block1, h, t_emb, use_reentrant=False)
        else:
            h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        if self.use_checkpoint:
            h = grad_checkpoint(self.mid_block2, h, t_emb, use_reentrant=False)
        else:
            h = self.mid_block2(h, t_emb)

        for blocks, sampler in zip(self.up_blocks, self.up_samplers):
            for block in blocks:
                if isinstance(block, TimeConditionedResBlock):
                    h = torch.cat([h, skips.pop()], dim=1)
                    if self.use_checkpoint:
                        h = grad_checkpoint(block, h, t_emb, use_reentrant=False)
                    else:
                        h = block(h, t_emb)
                else:
                    h = block(h)
            if sampler is not None:
                h = sampler(h)

        h = F.silu(self.out_norm(h))
        out = self.conv_out(h)
        if pad_h or pad_w:
            out = out[..., :h0, :w0]
        return out
