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

    num_feat: int = 32
    num_block: int = 6
    embed_dim: int = 64
    cond_dim: int = 64
    d_state: int = 16
    ssm_expand: int = 2
    sr_scale: int = 1

    refiner_base: int = 48
    channel_mult: Sequence[int] = (1, 2, 3)
    num_res_blocks: int = 2

    num_timesteps: int = 1000
    schedule: str = "cosine"
    refine_steps: int = 8

    # Min-SNR-gamma weighting of the diffusion loss (0 = off, 5.0 = paper value)
    min_snr_gamma: float = 5.0

    # The diffusion residual (gt - coarse) is normalised by a running estimate
    # of its own standard deviation before entering the noise schedule, and
    # scaled back out at sampling time. A FIXED scale does not work here: the
    # coarse branch keeps improving, so the residual keeps shrinking, and any
    # constant tuned today drifts into a degenerate target (v becomes a closed
    # form of x_t, the loss collapses to ~0 and the refiner learns nothing).
    # `residual_std_init` only seeds the buffer before the first train step;
    # during `residual_std_warmup` iterations the raw batch std is used, so the
    # scale locks on immediately while the coarse output is still random.
    residual_std_init: float = 0.15
    residual_std_momentum: float = 0.99
    residual_std_warmup: int = 200
    residual_std_min: float = 1e-3

    hole_threshold: float = 0.5

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
