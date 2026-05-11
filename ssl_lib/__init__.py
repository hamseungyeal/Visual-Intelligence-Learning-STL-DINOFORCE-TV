"""ssl_lib — Self-Supervised Learning library."""
__version__ = "0.1.0"

from .train_loop import pretrain, train_one_epoch, build_model

__all__ = ["pretrain", "train_one_epoch", "build_model"]
