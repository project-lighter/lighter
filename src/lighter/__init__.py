"""
Lighter is a framework for streamlining deep learning experiments with configuration files.
"""

__version__ = "0.0.5"

from .utils.logging import _setup_logging

_setup_logging()

from .engine.runner import Runner  # noqa: E402
from .system import System  # noqa: E402

__all__ = ["Runner", "System"]
