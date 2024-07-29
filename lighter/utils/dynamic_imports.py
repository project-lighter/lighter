from typing import Dict

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from monai.utils.module import optional_import


@dataclass
class OptionalImports:
    """Dataclass for handling optional imports.

    This class provides a way to handle optional imports in a convenient manner.
    It allows importing modules that may or may not be available, and raises an ImportError if the module is not available.

    Example:

        from lighter.utils.dynamic_imports import OPTIONAL_IMPORTS

        writer = OPTIONAL_IMPORTS["tensorboard"].SummaryWriter()

    Attributes:
        imports (Dict[str, object]): A dictionary to store the imported modules.
    """

    imports: Dict[str, object] = field(default_factory=dict)

    def __getitem__(self, module_name: str) -> "module":
        """Get the imported module by name.

        Args:
            module_name (str): Name of the module to import.

        Raises:
            ImportError: If the module is not available.

        Returns:
            Imported module.
        """
        if module_name not in self.imports:
            self.imports[module_name], module_available = optional_import(module_name)
            if not module_available:
                raise ImportError(f"'{module_name}' is not available. Make sure that it is installed and spelled correctly.")
        return self.imports[module_name]


OPTIONAL_IMPORTS = OptionalImports()


def import_module_from_path(module_name: str, module_path: str) -> None:
    """Import a module from a given path and assign it a specified name.

    This function imports a module from the specified path and assigns it the specified name.

    Args:
        module_name (str): Name to assign to the imported module.
        module_path (str): Path to the module being imported.

    Raises:
        ValueError: If the module has already been imported.
        FileNotFoundError: If the `__init__.py` file is not found in the module path.
    """
    # Based on https://stackoverflow.com/a/41595552.

    if module_name in sys.modules:
        logger.warning(f"{module_name} has already been imported as module.")
        return

    module_path = Path(module_path).resolve() / "__init__.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"No `__init__.py` in `{module_path}`.")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    logger.info(f"{module_path.parent} imported as '{module_name}' module.")
