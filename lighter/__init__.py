"""
Lighter is a framework for streamlining deep learning experiments with configuration files.
"""

__version__ = "0.0.3a11"

from .utils.logging import _setup_logging

_setup_logging()

# Expose Trainer and Tuner from PyTorch Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.tuner import Tuner

# Expose the System
from .system import LighterSystem
