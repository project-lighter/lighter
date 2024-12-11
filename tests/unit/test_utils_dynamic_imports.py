import sys
from unittest.mock import MagicMock, patch

import pytest

from lighter.utils.dynamic_imports import OPTIONAL_IMPORTS, import_module_from_path


def test_import_module_from_path():
    with pytest.raises(FileNotFoundError):
        import_module_from_path("non_existent_module", "non_existent_path")


def test_optional_imports():
    # Test importing a module that is not available
    with pytest.raises(ImportError):
        _ = OPTIONAL_IMPORTS["non_existent_module"]

    # Test importing a module that is available
    with patch("lighter.utils.dynamic_imports.optional_import", return_value=(MagicMock(), True)):
        module = OPTIONAL_IMPORTS["existent_module"]
        assert module is not None


def test_import_module_from_path_already_imported():
    # Mock sys.modules to simulate a module already imported
    with patch.dict(sys.modules, {"already_imported_module": MagicMock()}):
        import_module_from_path("already_imported_module", "some_path")
        # No exception should be raised, and the module should remain in sys.modules
        assert "already_imported_module" in sys.modules


def test_import_module_from_path_with_init():
    # Mock the Path and importlib to simulate a valid module path
    with patch("lighter.utils.dynamic_imports.Path") as mock_path, patch(
        "lighter.utils.dynamic_imports.importlib.util.spec_from_file_location"
    ) as mock_spec, patch("lighter.utils.dynamic_imports.importlib.util.module_from_spec") as mock_module:
        mock_path.return_value.resolve.return_value.__truediv__.return_value.is_file.return_value = True
        mock_spec.return_value = MagicMock()
        mock_module.return_value = MagicMock()

        import_module_from_path("valid_module", "valid_path")
        # Check that the module is added to sys.modules
        assert "valid_module" in sys.modules
