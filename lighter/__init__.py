"""
Lighter is a framework for streamlining deep learning experiments with configuration files.
"""

__version__ = "0.0.3a11"

from .utils.logging import _setup_logging

_setup_logging()

# Expose Trainer from PyTorch Lightning for convenience
from pytorch_lightning import Trainer

from .engine import Config, Runner
from .system import System
