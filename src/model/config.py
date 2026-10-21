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
    raft: str = "sintel"        # optical flow: sintel (MambaOFR's) | large | small
    raft_iters: int = 24        # RAFT refinement iterations (MambaOFR: 24)

    # Reconstruction tail
    sr_scale: int = 2           # SR factor (PixelShuffle)
    tail_feat: int = 64         # tail width; 16 = MambaOFR's tail

    # Phase 2: hole detection / flow completion / inpainting. Set by the
    # trainer from training.stage and stored with the checkpoint's model_config.
    use_holes: bool = False
    inpaint_dim: int = 64       # width of the inpainter (MambaIR embed_dim)
    inpaint_depths: Sequence[int] = (2, 2, 2)
    hole_threshold: float = 0.9      # detector probability -> hole (input pre-fill)
    inpaint_dilate: int = 4          # inpainter region = hole probability dilated by this (LR px)
    hole_prob: float = 0.25          # prob. a training window gets holes burned in

    @classmethod
    def from_dict(cls, d: dict | None) -> "ModelConfig":
        kwargs = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        for key in ("depths", "inpaint_depths"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)
