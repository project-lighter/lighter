"""
Lighter is a framework for streamlining deep learning experiments with configuration files.
"""

__version__ = "0.2.0.dev0"

from .data import LighterDataModule
from .engine.runner import Runner
from .model import LighterModule

__all__ = ["LighterDataModule", "LighterModule", "Runner"]
