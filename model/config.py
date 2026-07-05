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

    num_feat: int = 16
    num_block: int = 6
    embed_dim: int = 64
    cond_dim: int = 64
    d_state: int = 16
    ssm_expand: int = 2
    sr_scale: int = 1

    refiner_base: int = 64
    channel_mult: Sequence[int] = (1, 2, 4)
    num_res_blocks: int = 2

    mask_embed_dim: int = 16

    num_timesteps: int = 1000
    schedule: str = "cosine"
    refine_steps: int = 8

    hole_threshold: float = 0.5

    vfi_prob: float = 0.5
    vfi_mask_ratio: float = 0.3

    hole_prob: float = 0.15

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
