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
    module_path = Path(module_path).resolve() / "__init__.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"No `__init__.py` in `{module_path}`.")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    logger.info(f"{module_path.parent} imported as '{module_name}' module.")
