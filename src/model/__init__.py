"""REVIVID — MambaOFR restoration + 2x SR + inpainting of persistent holes."""

from .backbone import Video_Backbone
from .config import ModelConfig

__all__ = ["ModelConfig", "Video_Backbone"]
