"""Model configuration — mirrors the ``model:`` section of ``config/REVIVID.yaml``.

Typed defaults for building the network; every tunable value lives in the YAML.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass
class ModelConfig:
    # Backbone = MambaOFR's
    num_feat: int = 16          # propagated state channels
    embed_dim: int = 64         # width of the Mamba restorer (MambaIR embed_dim)
    depths: Sequence[int] = (2, 2, 2)  # VSS blocks per residual group
    d_state: int = 16           # SSM state dimension
    ssm_expand: float = 1.2     # SS2D inner expansion (MambaOFR's mlp_ratio)
    dyn_kernels: int = 8        # kernels of the degradation-aware prompt convs
    drop_path: float = 0.1      # stochastic depth of the VSS blocks
    cond_dim: int = 64          # backbone features handed to the refiner
    sr_scale: int = 2           # SR factor of the tail (PixelShuffle)
    raft: str = "sintel"        # optical flow: sintel (MambaOFR's) | large | small
    raft_iters: int = 24        # RAFT refinement iterations (MambaOFR: 24)
    # Phase 2: hole detection / filling / flow completion. Set by the trainer
    # from training.stage and stored with the checkpoint's model_config.
    use_holes: bool = False

    # Refiner (diffusion UNet, inside holes only)
    refiner_base: int = 48
    channel_mult: Sequence[int] = (1, 2, 3)
    num_res_blocks: int = 2
    refiner_temporal_levels: Sequence[int] = (1, 2)  # attention across frames
    num_timesteps: int = 1000
    refine_steps: int = 8       # DDIM steps at inference
    min_snr_gamma: float = 5.0

    # The diffusion works on (gt - coarse) divided by a running estimate of its
    # std inside the holes, so the target stays in the schedule's ~N(0, 1)
    # range however good `coarse` gets.
    residual_std_init: float = 0.15
    residual_std_momentum: float = 0.99
    residual_std_warmup: int = 200   # updates using the raw batch std
    residual_std_min: float = 1e-3

    # Holes
    hole_threshold: float = 0.9      # detector probability -> hole
    refine_mask_dilate: int = 8      # refiner region = holes dilated by this (HR px)
    hole_prob: float = 0.25          # prob. a training window gets holes burned in

    @classmethod
    def from_dict(cls, d: dict | None) -> "ModelConfig":
        kwargs = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        for key in ("depths", "channel_mult", "refiner_temporal_levels"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)
