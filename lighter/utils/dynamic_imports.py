"""
This module provides utilities for dynamic imports, allowing optional imports and importing modules from paths.
"""

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from monai.utils.module import optional_import


@dataclass
class OptionalImports:
    """
    Handles optional imports, allowing modules to be imported only if they are available.

    Attributes:
        imports: A dictionary to store the imported modules.

    Example:
        ```
        from lighter.utils.dynamic_imports import OPTIONAL_IMPORTS
        writer = OPTIONAL_IMPORTS["tensorboard"].SummaryWriter()
        ```
    """

    imports: dict[str, object] = field(default_factory=dict)

    def __getitem__(self, module_name: str) -> object:
        """
        Get the imported module by name, importing it if necessary.

        Args:
            module_name: Name of the module to import.

        Raises:
            ImportError: If the module is not available.

        Returns:
            object: The imported module.
        """
        """Get the imported module by name.

        Args:
            module_name: Name of the module to import.

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
    """
    Import a module from a given path and assign it a specified name.

    Args:
        module_name: Name to assign to the imported module.
        module_path: Path to the module being imported.

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
    logger.info(f"Imported {module_path.parent} as module '{module_name}'.")
