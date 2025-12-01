"""Neural network architectures for self-supervised learning."""

from .encoder import BYOLNetwork, SimCLRNetwork, create_byol_model, create_simclr_model

__all__ = ["SimCLRNetwork", "BYOLNetwork", "create_simclr_model", "create_byol_model"]
