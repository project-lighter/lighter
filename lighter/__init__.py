__version__ = "0.0.2a26"

from .utils.logging import _setup_logging

_setup_logging()

from .system import LighterSystem
