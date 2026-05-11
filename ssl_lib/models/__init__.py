"""SSL model architectures."""
from .backbone import build_backbone, ResNetBackbone
from .heads import ProjectionHead, Predictor
from .mocov2 import MoCoV2
from .byol import BYOL

__all__ = [
    "build_backbone",
    "ResNetBackbone",
    "ProjectionHead",
    "Predictor",
    "MoCoV2",
    "BYOL",
]
