"""Model-architecture configuration for the REVIVID unified MFP network.

This dataclass mirrors the ``model:`` section of ``config/REVIVID.yaml``. It is
model code (typed defaults for building the network), not a user-facing config
file - all tunable values live in the YAML.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass
class ModelConfig:
    # Backbone
    num_feat: int = 16
    num_block: int = 6
    embed_dim: int = 64
    cond_dim: int = 64
    d_state: int = 16
    ssm_expand: int = 2
    sr_scale: int = 1

    # Diffusion U-Net (single unified denoiser)
    refiner_base: int = 64
    channel_mult: Sequence[int] = (1, 2, 4)
    num_res_blocks: int = 2
    # Embedding dim for temporal frame-mask signal fed into the UNet conditioning.
    # Small value (16) is enough to distinguish observed vs masked positions.
    mask_embed_dim: int = 16

    # Diffusion process
    num_timesteps: int = 1000
    schedule: str = "cosine"
    refine_steps: int = 8

    # Persistent spatial holes: detected on-the-fly from LQ fill value
    hole_threshold: float = 0.5   # sigmoid threshold for hole_head at inference

    # VFI (temporal masked frame prediction)
    vfi_prob: float = 0.5         # probability a training clip has masked frames
    vfi_mask_ratio: float = 0.3   # fraction of inner frames that may be masked

    # Persistent spatial holes (applied per window by DatasetCreator / DataLoader)
    hole_prob: float = 0.15       # probability a training window has holes burned in

    @classmethod
    def from_dict(cls, d: dict | None) -> "ModelConfig":
        if not d:
            return cls()
        fields = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in d.items() if k in fields}
        if kwargs.get("channel_mult") is not None:
            kwargs["channel_mult"] = tuple(kwargs["channel_mult"])
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)
