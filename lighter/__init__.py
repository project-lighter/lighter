"""
Lighter is a framework for streamlining deep learning experiments with configuration files.
"""

__version__ = "0.0.3a21"

from .utils.logging import _setup_logging

_setup_logging()

from .engine.config import Config
from .engine.runner import Runner
from .system import System
