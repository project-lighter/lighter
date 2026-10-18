"""Tests for dynamic_imports module."""

import importlib
import pickle
import sys
from io import BytesIO
from unittest.mock import patch

import pytest

from lighter.utils.dynamic_imports import _HybridPickler, import_module_from_path


def _dumps(obj):
    buffer = BytesIO()
    _HybridPickler(buffer).dump(obj)
    return buffer.getvalue()


class TestImportModuleFromPath:
    """Tests for import_module_from_path function."""

    def test_nonexistent_path_raises_error(self):
        """Importing from a folder that does not exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No folder"):
            import_module_from_path("nonexistent", "/path/that/does/not/exist")

    def test_folder_without_init_is_imported_as_a_package(self, tmp_path):
        """`__init__.py` is optional: submodules of a plain folder import and pickle by value."""
        (tmp_path / "models.py").write_text("def double(x):\n    return 2 * x\n")
        try:
            module = import_module_from_path("no_init_project", tmp_path)
            assert list(module.__path__) == [str(tmp_path)]
            pickled = _dumps(importlib.import_module("no_init_project.models").double)
        finally:
            sys.modules.pop("no_init_project", None)
            sys.modules.pop("no_init_project.models", None)
        assert pickle.loads(pickled)(3) == 6  # rebuilt without importing the project

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

    def test_reducer_override_defers_multiprocessing_objects(self):
        """reducer_override returns NotImplemented for multiprocessing internals.

        Multiprocessing objects (Queue, Pipe connections, etc.) have special reducers
        in ForkingPickler._extra_reducers that must be preserved. HybridPickler's
        reducer_override should return NotImplemented for these objects, allowing
        the standard ForkingPickler dispatch to handle them correctly.
        """
        import multiprocessing
        from multiprocessing.connection import Connection

        buffer = BytesIO()
        pickler = _HybridPickler(buffer)

        # Create a Pipe which gives us Connection objects - these are in _extra_reducers
        recv_conn, send_conn = multiprocessing.Pipe(duplex=False)

        try:
            # Verify Connection type is in _extra_reducers
            from multiprocessing.reduction import ForkingPickler

            extra_reducers = getattr(ForkingPickler, "_extra_reducers", {})
            assert Connection in extra_reducers, "Connection should be in _extra_reducers"

            # reducer_override should return NotImplemented for Connection objects
            result = pickler.reducer_override(recv_conn)
            assert result is NotImplemented, "Should defer to ForkingPickler for Connection"

            result = pickler.reducer_override(send_conn)
            assert result is NotImplemented, "Should defer to ForkingPickler for Connection"

            # Verify the object can still be pickled using HybridPickler
            # (dispatch_table includes ForkingPickler's reducers)
            buffer = BytesIO()
            pickler = _HybridPickler(buffer)
            pickler.dump(send_conn)

            # Verify serialization produced data
            assert buffer.tell() > 0, "Should have written pickle data"

        finally:
            recv_conn.close()
            send_conn.close()

    def test_multiprocessing_objects_work_through_subprocess(self):
        """Verify multiprocessing objects can be passed through actual child processes.

        This test ensures that the ForkingPickler._extra_reducers handling remains intact
        by actually passing a multiprocessing Queue through a spawn-started child process.
        The Queue uses Connection objects internally which are in _extra_reducers.
        """
        import multiprocessing

        # Use spawn context to match PyTorch DataLoader behavior
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()

        # Define worker function outside to avoid pickling issues
        def worker(q):
            q.put("success")

        # Start a child process that uses the queue
        process = ctx.Process(target=worker, args=(queue,))
        process.start()
        process.join(timeout=10)

        # Verify the queue worked correctly through the subprocess
        assert not queue.empty(), "Queue should have received data from child process"
        result = queue.get(timeout=1)
        assert result == "success", "Should receive correct data from child process"

        # Cleanup
        queue.close()
        queue.join_thread()


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

    def test_module_load_failure_cleans_up_state(self, tmp_path):
        """Test that module load failure cleans up sys.modules."""
        # Create a package with a syntax error in __init__.py
        pkg_dir = tmp_path / "broken_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("def broken(\n")  # Syntax error

        module_name = "test_broken_module_cleanup"

        # Verify module is not in sys.modules before
        assert module_name not in sys.modules

        # Attempt import - should fail with SyntaxError
        with pytest.raises(SyntaxError):
            import_module_from_path(module_name, pkg_dir)

        # Verify cleanup: module should NOT be in sys.modules after failure
        assert module_name not in sys.modules

    def test_module_runtime_error_cleans_up_state(self, tmp_path):
        """Test that runtime error during module execution cleans up sys.modules.

        This test triggers a module load failure via a runtime exception (not syntax error)
        to verify that a failed import removes the module from sys.modules.
        """

        # Create a package that raises an error during execution
        pkg_dir = tmp_path / "runtime_error_pkg"
        pkg_dir.mkdir()
        init_content = """
# This module raises an error during import
VALUE = 1
raise RuntimeError("Intentional failure during module load")
"""
        (pkg_dir / "__init__.py").write_text(init_content)

        module_name = "test_runtime_error_module_cleanup"

        # Capture initial state
        initial_modules = set(sys.modules.keys())
        assert module_name not in sys.modules, "Module should not exist before test"

        # Attempt import - should fail with RuntimeError
        with pytest.raises(RuntimeError, match="Intentional failure"):
            import_module_from_path(module_name, pkg_dir)

        # Verify cleanup: module should NOT be in sys.modules after failure
        assert module_name not in sys.modules, "Failed module must be removed from sys.modules"

        # Verify no new modules were left behind (check module and potential submodules)
        current_modules = set(sys.modules.keys())
        new_modules = current_modules - initial_modules
        assert not any(m.startswith(module_name) for m in new_modules), (
            f"No modules starting with '{module_name}' should remain, but found: "
            f"{[m for m in new_modules if m.startswith(module_name)]}"
        )

    def test_module_import_error_cleans_up_state(self, tmp_path):
        """Test that ImportError during module execution cleans up sys.modules.

        This test triggers a failure via a missing import to verify cleanup works for
        ImportError exceptions as well.
        """

        # Create a package that fails due to missing import
        pkg_dir = tmp_path / "import_error_pkg"
        pkg_dir.mkdir()
        init_content = """
# This module fails to import a nonexistent module
from nonexistent_module_xyz_12345 import something
"""
        (pkg_dir / "__init__.py").write_text(init_content)

        module_name = "test_import_error_module_cleanup"

        # Verify preconditions
        assert module_name not in sys.modules

        # Attempt import - should fail with ModuleNotFoundError
        with pytest.raises(ModuleNotFoundError):
            import_module_from_path(module_name, pkg_dir)

        # Verify cleanup
        assert module_name not in sys.modules, "Failed module must be removed from sys.modules"


@pytest.mark.slow
def test_project_classes_work_in_spawned_dataloader_workers(tmp_path):
    """A dataset defined in an imported project folder can be sent to spawned DataLoader workers."""
    from torch.utils.data import DataLoader

    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "data.py").write_text(
        "import torch\n"
        "class Squares(torch.utils.data.Dataset):\n"
        "    def __len__(self):\n"
        "        return 6\n"
        "    def __getitem__(self, i):\n"
        "        return torch.tensor(i * i)\n"
    )
    module_name = "spawn_worker_project"
    try:
        import_module_from_path(module_name, tmp_path)
        data = importlib.import_module(f"{module_name}.data")
        loader = DataLoader(data.Squares(), batch_size=3, num_workers=1, multiprocessing_context="spawn")
        assert [batch.tolist() for batch in loader] == [[0, 1, 4], [9, 16, 25]]
    finally:
        sys.modules.pop(module_name, None)
        sys.modules.pop(f"{module_name}.data", None)
