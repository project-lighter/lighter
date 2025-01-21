"""
Lighter is a framework for streamlining deep learning experiments with configuration files.
"""

from .system import LighterSystem

__version__ = "0.0.3a11"

from .utils.logging import _setup_logging

_setup_logging()
