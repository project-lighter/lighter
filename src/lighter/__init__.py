"""
Lighter is a framework for streamlining deep learning experiments with configuration files.
"""

__version__ = "0.1.0"

from .utils.logging import _setup_logging

_setup_logging()

from .data import LighterDataModule  # noqa: E402
from .engine.runner import Runner  # noqa: E402
from .model import LighterModule  # noqa: E402

__all__ = ["LighterDataModule", "LighterModule", "Runner"]
