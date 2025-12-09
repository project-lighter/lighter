"""Dynamic module imports with multiprocessing spawn support.

Enables importing Python packages from arbitrary paths while maintaining compatibility
with multiprocessing spawn workers (used by PyTorch DataLoader with num_workers > 0).

Uses cloudpickle to serialize dynamically imported modules by value, embedding class
definitions in the pickle stream rather than storing module paths that workers can't resolve.

The key insight is that we need cloudpickle for user-defined classes (to serialize them
by value), but we must preserve ForkingPickler's special reducers for multiprocessing
internals (pipes, queues, connections, file descriptors). We achieve this by creating
a hybrid pickler that inherits from ForkingPickler but also includes cloudpickle's
dispatch table.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from collections.abc import Sequence
from io import BytesIO
from multiprocessing import reduction
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from types import ModuleType
from typing import IO, Any, cast

import cloudpickle

__all__ = ["import_module_from_path"]


class _ModuleRegistry:
    """Maps dynamically imported module names to their filesystem paths."""

    def __init__(self) -> None:
        self._modules: dict[str, Path] = {}

    def register(self, name: str, path: Path) -> None:
        self._modules[name] = path

    def get(self, name: str) -> Path | None:
        """Get the registered path for a module name."""
        return self._modules.get(name)

    def find_root(self, fullname: str) -> tuple[str, Path] | None:
        """Find the registered root module for an import name (e.g., 'project.sub' -> 'project')."""
        for name, path in self._modules.items():
            if fullname == name or fullname.startswith(f"{name}."):
                return name, path
        return None


_registry = _ModuleRegistry()


class _DynamicModuleFinder(importlib.abc.MetaPathFinder):
    """Meta path finder for submodules of dynamically imported packages."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str | bytes] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        root = _registry.find_root(fullname)
        if root is None:
            return None

        root_name, root_path = root
        file_path = self._resolve_path(fullname, root_name, root_path)
        if file_path is None or not file_path.is_file():
            return None

        return importlib.machinery.ModuleSpec(
            fullname,
            _DynamicModuleLoader(str(file_path), fullname),
            origin=str(file_path),
            is_package=file_path.name == "__init__.py",
        )

    def _resolve_path(self, fullname: str, root_name: str, root_path: Path) -> Path | None:
        """Resolve 'project.models.net' to '/path/to/project/models/net.py'."""
        if fullname == root_name:
            return root_path / "__init__.py"

        relative = fullname[len(root_name) + 1 :].replace(".", "/")

        # Try package first, then module
        package_init = root_path / relative / "__init__.py"
        if package_init.is_file():
            return package_init
        return root_path / f"{relative}.py"


class _DynamicModuleLoader(importlib.abc.Loader):
    """Loader that registers modules with cloudpickle for by-value serialization."""

    def __init__(self, filepath: str, fullname: str) -> None:
        self._filepath = filepath
        self._fullname = fullname

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        importlib.machinery.SourceFileLoader(self._fullname, self._filepath).exec_module(module)
        cloudpickle.register_pickle_by_value(module)


class _HybridPickler(ForkingPickler):
    """Pickler combining ForkingPickler's reducers with cloudpickle's by-value serialization.

    ForkingPickler has special reducers for multiprocessing internals (pipes, queues,
    connections, file descriptors) that must be preserved. cloudpickle can serialize
    dynamically defined classes by value. This hybrid uses both: ForkingPickler's
    dispatch table takes priority (for multiprocessing internals), then cloudpickle's
    reducer_override handles user-defined classes.
    """

    def __init__(self, file: IO[bytes], protocol: int | None = None) -> None:
        super().__init__(file, protocol)
        # Merge cloudpickle's dispatch into our dispatch_table
        # ForkingPickler's reducers (from _extra_reducers) take priority
        cloudpickle_dispatch = cloudpickle.CloudPickler.dispatch.copy()
        cloudpickle_dispatch.update(self.dispatch_table)
        self.dispatch_table = cloudpickle_dispatch

    def reducer_override(self, obj: Any) -> Any:
        """Use cloudpickle's reducer for objects not in dispatch_table."""
        # If it's in ForkingPickler's extra reducers, let standard dispatch handle it
        # _extra_reducers is a class attribute that exists at runtime but isn't typed
        extra_reducers = cast(dict[type, Any], getattr(ForkingPickler, "_extra_reducers", {}))
        if type(obj) in extra_reducers:
            return NotImplemented

        # For everything else, try cloudpickle's reducer_override
        # This handles dynamically imported classes registered with register_pickle_by_value
        pickler = cloudpickle.CloudPickler(BytesIO())
        return pickler.reducer_override(obj)


def _hybrid_dump(obj: Any, file: IO[bytes], protocol: int | None = None) -> None:
    """Replacement for reduction.dump using hybrid pickler."""
    _HybridPickler(file, protocol).dump(obj)


# Module initialization: install finder and patch multiprocessing
sys.meta_path.insert(0, _DynamicModuleFinder())
reduction.dump = _hybrid_dump  # type: ignore[assignment]


def import_module_from_path(module_name: str, module_path: Path | str) -> ModuleType:
    """Import a package from a filesystem path with multiprocessing support.

    Args:
        module_name: Name to assign to the module (e.g., "project").
        module_path: Path to the package directory (must contain __init__.py).

    Returns:
        The imported module.

    Raises:
        FileNotFoundError: If module_path doesn't contain __init__.py.
        ModuleNotFoundError: If the module cannot be loaded.
        ValueError: If module_name was already imported from a different path.

    Example:
        >>> import_module_from_path("project", "/path/to/project")
        >>> from project.models import MyModel  # Works in DataLoader workers!
    """
    module_path = Path(module_path).resolve()

    # Check if already imported
    if module_name in sys.modules:
        existing_path = _registry.get(module_name)
        if existing_path is not None and existing_path != module_path:
            raise ValueError(f"Module '{module_name}' was already imported from '{existing_path}'.")
        # Same path - return cached module (normal Python behavior)
        return sys.modules[module_name]

    init_file = module_path / "__init__.py"

    if not init_file.is_file():
        raise FileNotFoundError(f"No __init__.py in '{module_path}'.")

    spec = importlib.util.spec_from_file_location(module_name, str(init_file))
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Could not load '{module_name}' from '{module_path}'.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        cloudpickle.register_pickle_by_value(module)
        _registry.register(module_name, module_path)
    except Exception:
        # Clean up on failure so retry sees a clean state
        sys.modules.pop(module_name, None)
        raise

    return module
