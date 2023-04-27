from typing import Any

import importlib
import sys
from pathlib import Path

from loguru import logger

OPTIONAL_IMPORTS = {}


def import_module_from_path(module_name: str, module_path: str) -> None:
    """Given the path to a module, import it, and name it as specified.

    Args:
        module_name (str): what to name the imported module.
        module_path (str): path to the module to load.
    """
    # Based on https://stackoverflow.com/a/41595552.

    if module_name in sys.modules:
        logger.error(f"{module_path} has already been imported as module: {module_name}")
        sys.exit()

    module_path = Path(module_path).resolve() / "__init__.py"
    if not module_path.is_file():
        logger.error(f"No `__init__.py` in `{module_path}`. Exiting.")
        sys.exit()
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    logger.info(f"{module_path.parent} imported as '{module_name}' module.")


def import_attr(module_attr: str) -> Any:
    """Import using dot-notation string, e.g., 'torch.nn.Module'.

    Args:
        module_attr (str): dot-notation path to the attribute.

    Returns:
        Any: imported attribute.
    """
    # Split module from attribute name
    module, attr = module_attr.rsplit(".", 1)
    # Import the module
    module = __import__(module, fromlist=[attr])
    # Get the attribute from the module
    return getattr(module, attr)
