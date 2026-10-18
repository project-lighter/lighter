"""Import a project folder as a package that also works in spawned DataLoader workers.

The imported package is registered with cloudpickle, so its classes and functions are pickled
by value and a spawned worker can rebuild them without importing the project. Once a project
is imported, multiprocessing uses a pickler that adds cloudpickle to ForkingPickler while
keeping ForkingPickler's own reducers for pipes, queues and file descriptors.
"""

import importlib.util
import sys
from io import BytesIO
from multiprocessing import reduction
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from types import ModuleType
from typing import IO, Any

import cloudpickle

__all__ = ["import_module_from_path"]


class _HybridPickler(ForkingPickler):
    """ForkingPickler whose remaining objects go through cloudpickle (by value for registered modules)."""

    def __init__(self, file: IO[bytes], protocol: int | None = None) -> None:
        super().__init__(file, protocol)
        dispatch_table = cloudpickle.CloudPickler.dispatch.copy()
        dispatch_table.update(self.dispatch_table)  # ForkingPickler's reducers take priority
        self.dispatch_table = dispatch_table

    def reducer_override(self, obj: Any) -> Any:
        if type(obj) in getattr(ForkingPickler, "_extra_reducers", {}):
            return NotImplemented
        return cloudpickle.CloudPickler(BytesIO()).reducer_override(obj)


def _hybrid_dump(obj: Any, file: IO[bytes], protocol: int | None = None) -> None:
    """Replacement for `multiprocessing.reduction.dump`."""
    _HybridPickler(file, protocol).dump(obj)


def import_module_from_path(module_name: str, module_path: Path | str) -> ModuleType:
    """Import the package in `module_path` under the name `module_name` (e.g. "project").

    Raises:
        FileNotFoundError: If module_path doesn't contain __init__.py.
        ValueError: If module_name was already imported from a different path.
    """
    module_path = Path(module_path).resolve()
    if module_name in sys.modules:
        existing = getattr(sys.modules[module_name], "__path__", None)
        if existing and Path(list(existing)[0]).resolve() != module_path:
            raise ValueError(f"Module '{module_name}' was already imported from '{list(existing)[0]}'.")
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
    except Exception:
        sys.modules.pop(module_name, None)  # so that a retry starts clean
        raise
    cloudpickle.register_pickle_by_value(module)
    reduction.dump = _hybrid_dump  # type: ignore[assignment]
    return module
