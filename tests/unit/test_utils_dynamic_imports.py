"""Tests for dynamic_imports module."""

import pickle
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from lighter.utils.dynamic_imports import (
    _DynamicModuleFinder,
    _HybridPickler,
    _ModuleRegistry,
    import_module_from_path,
)


class TestImportModuleFromPath:
    """Tests for import_module_from_path function."""

    def test_nonexistent_path_raises_error(self):
        """Importing from a path without __init__.py raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No __init__.py"):
            import_module_from_path("nonexistent", "/path/that/does/not/exist")

    def test_already_imported_same_path_returns_existing(self, tmp_path):
        """If module is already imported from same path, return cached module."""

        # Create a real package
        pkg_dir = tmp_path / "test_cached_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("VALUE = 99")

        module_name = "test_cached_module"
        try:
            # First import
            result1 = import_module_from_path(module_name, pkg_dir)
            # Second import from same path should return cached
            result2 = import_module_from_path(module_name, pkg_dir)
            assert result1 is result2
        finally:
            sys.modules.pop(module_name, None)

    def test_already_imported_different_path_raises_error(self, tmp_path):
        """If module is already imported from different path, raise ValueError."""
        # Create two different packages
        pkg_dir1 = tmp_path / "pkg1"
        pkg_dir1.mkdir()
        (pkg_dir1 / "__init__.py").write_text("VALUE = 1")

        pkg_dir2 = tmp_path / "pkg2"
        pkg_dir2.mkdir()
        (pkg_dir2 / "__init__.py").write_text("VALUE = 2")

        module_name = "test_conflict_module"
        try:
            # First import
            import_module_from_path(module_name, pkg_dir1)
            # Second import from different path should raise
            with pytest.raises(ValueError, match="already imported from"):
                import_module_from_path(module_name, pkg_dir2)
        finally:
            sys.modules.pop(module_name, None)

    def test_successful_import(self, tmp_path):
        """Successfully import a real package from filesystem."""
        # Create a real package
        pkg_dir = tmp_path / "test_real_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("VALUE = 42")

        module_name = "test_real_pkg_import"
        try:
            result = import_module_from_path(module_name, pkg_dir)

            assert result is not None
            assert result.VALUE == 42
            assert module_name in sys.modules
        finally:
            # Cleanup
            sys.modules.pop(module_name, None)

    def test_import_with_submodule(self, tmp_path):
        """Import a package and verify submodules can be imported."""
        # Create package with submodule
        pkg_dir = tmp_path / "test_pkg_with_sub"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "submod.py").write_text("SUBVALUE = 123")

        module_name = "test_pkg_with_sub_import"
        try:
            import_module_from_path(module_name, pkg_dir)

            # Import submodule using standard import
            import importlib

            submod = importlib.import_module(f"{module_name}.submod")
            assert submod.SUBVALUE == 123
        finally:
            # Cleanup
            sys.modules.pop(module_name, None)
            sys.modules.pop(f"{module_name}.submod", None)


class TestModuleRegistry:
    """Tests for _ModuleRegistry."""

    def test_register_and_find_exact_match(self):
        """Registry returns exact match for registered module."""
        registry = _ModuleRegistry()
        test_path = Path("/test/path")
        registry.register("mymodule", test_path)

        result = registry.find_root("mymodule")
        assert result == ("mymodule", test_path)

    def test_find_submodule_returns_root(self):
        """Registry returns root module for submodule queries."""
        registry = _ModuleRegistry()
        test_path = Path("/test/path")
        registry.register("mymodule", test_path)

        result = registry.find_root("mymodule.sub.deep")
        assert result == ("mymodule", test_path)

    def test_find_unregistered_returns_none(self):
        """Registry returns None for unregistered modules."""
        registry = _ModuleRegistry()
        assert registry.find_root("unregistered") is None


class TestDynamicModuleFinder:
    """Tests for _DynamicModuleFinder."""

    def test_unregistered_module_returns_none(self):
        """Finder returns None for modules not in registry."""
        finder = _DynamicModuleFinder()
        # Use a unique name that won't be registered
        result = finder.find_spec("completely_unknown_module_xyz", None, None)
        assert result is None

    def test_missing_file_returns_none(self, tmp_path):
        """Finder returns None when submodule file doesn't exist."""
        from lighter.utils.dynamic_imports import _registry

        # Create package without the submodule we'll look for
        pkg_dir = tmp_path / "finder_test_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")

        module_name = "finder_test_pkg_missing"
        _registry.register(module_name, pkg_dir)

        try:
            finder = _DynamicModuleFinder()
            result = finder.find_spec(f"{module_name}.nonexistent", None, None)
            assert result is None
        finally:
            # Registry doesn't have unregister, but module names are unique per test
            pass

    def test_finds_root_package(self, tmp_path):
        """Finder creates spec for root package __init__.py."""
        from lighter.utils.dynamic_imports import _registry

        pkg_dir = tmp_path / "finder_root_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("# root")

        module_name = "finder_root_pkg_test"
        _registry.register(module_name, pkg_dir)

        finder = _DynamicModuleFinder()
        result = finder.find_spec(module_name, None, None)

        assert result is not None
        assert result.name == module_name
        assert "__init__.py" in result.origin

    def test_finds_submodule(self, tmp_path):
        """Finder creates spec for submodule .py file."""
        from lighter.utils.dynamic_imports import _registry

        pkg_dir = tmp_path / "finder_sub_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "child.py").write_text("X = 1")

        module_name = "finder_sub_pkg_test"
        _registry.register(module_name, pkg_dir)

        finder = _DynamicModuleFinder()
        result = finder.find_spec(f"{module_name}.child", None, None)

        assert result is not None
        assert result.name == f"{module_name}.child"
        assert "child.py" in result.origin


class TestHybridPickler:
    """Tests for _HybridPickler."""

    def test_pickles_basic_types(self):
        """HybridPickler can pickle and unpickle basic Python types."""
        buffer = BytesIO()
        pickler = _HybridPickler(buffer)

        data = {"key": [1, 2, 3], "nested": {"a": "b"}}
        pickler.dump(data)

        buffer.seek(0)
        result = pickle.load(buffer)
        assert result == data

    def test_pickles_lambda(self):
        """HybridPickler can pickle lambdas (via cloudpickle)."""
        buffer = BytesIO()
        pickler = _HybridPickler(buffer)

        fn = lambda x: x * 2  # noqa: E731
        pickler.dump(fn)

        buffer.seek(0)
        result = pickle.load(buffer)
        assert result(5) == 10

    def test_reducer_override_handles_functions(self):
        """reducer_override returns valid reduction for functions."""
        buffer = BytesIO()
        pickler = _HybridPickler(buffer)

        fn = lambda x: x + 1  # noqa: E731
        result = pickler.reducer_override(fn)

        # Should return a reduction tuple (callable, args) or similar
        # NotImplemented means "use default pickling"
        assert result is not NotImplemented


class TestImportModuleFromPathErrors:
    """Tests for error cases in import_module_from_path."""

    def test_spec_from_file_returns_none(self, tmp_path):
        """Test ModuleNotFoundError when spec_from_file_location returns None."""
        import importlib.util

        # Create a valid package
        pkg_dir = tmp_path / "test_spec_none_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("VALUE = 1")

        module_name = "test_spec_none_module"

        # Mock spec_from_file_location to return None
        with patch.object(importlib.util, "spec_from_file_location", return_value=None):
            with pytest.raises(ModuleNotFoundError, match="Could not load"):
                import_module_from_path(module_name, pkg_dir)
