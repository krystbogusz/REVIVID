"""REVIVID — restoration + 2x SR + inpainting of persistent holes."""

from .config import ModelConfig
from .video_diffusion_model import Video_Backbone

__all__ = ["ModelConfig", "Video_Backbone"]
