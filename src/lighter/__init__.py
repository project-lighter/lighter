"""Lighter: run Lightning stages from YAML configs."""

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"
__all__ = ["LighterDataModule", "LighterModule", "Runner", "build"]
_MODULES = {"LighterDataModule": "data", "LighterModule": "model", "Runner": "engine.runner", "build": "engine.runner"}

if TYPE_CHECKING:
    from .data import LighterDataModule
    from .engine.runner import Runner, build
    from .model import LighterModule


def __getattr__(name: str) -> Any:
    """Import the public names on first use, so the CLI starts without importing torch."""
    if name not in _MODULES:
        raise AttributeError(f"module 'lighter' has no attribute '{name}'")
    return getattr(importlib.import_module(f"lighter.{_MODULES[name]}"), name)
