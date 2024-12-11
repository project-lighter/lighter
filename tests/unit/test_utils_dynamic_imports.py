import sys
from unittest.mock import MagicMock, patch

import pytest

from lighter.utils.dynamic_imports import OPTIONAL_IMPORTS, import_module_from_path


def test_import_module_from_path_nonexistent():
    """
    Test importing a module from a nonexistent path raises FileNotFoundError.

    This test verifies that attempting to import a module from a path that
    doesn't exist results in a FileNotFoundError being raised.

    Raises:
        FileNotFoundError: Expected to be raised when path doesn't exist
    """
    with pytest.raises(FileNotFoundError):
        import_module_from_path("non_existent_module", "non_existent_path")


def test_optional_imports_nonexistent():
    """
    Test accessing a nonexistent module from OPTIONAL_IMPORTS raises ImportError.

    This test verifies that attempting to access a module that isn't defined
    in the OPTIONAL_IMPORTS dictionary raises an ImportError.

    Raises:
        ImportError: Expected to be raised when accessing undefined module
    """
    with pytest.raises(ImportError):
        _ = OPTIONAL_IMPORTS["non_existent_module"]


def test_optional_imports_available():
    """
    Test successful retrieval of an available optional import.

    This test verifies that when a module exists in OPTIONAL_IMPORTS, it:
    1. Returns the correct mock module
    2. Calls optional_import with the correct module name
    3. Only calls optional_import once

    Setup:
        - Creates a mock module
        - Patches optional_import to return the mock module
    """
    mock_module = MagicMock()
    with patch("lighter.utils.dynamic_imports.optional_import", return_value=(mock_module, True)) as mock_import:
        module = OPTIONAL_IMPORTS["existent_module"]
        assert module is mock_module
        mock_import.assert_called_once_with("existent_module")


def test_import_module_from_path_already_imported():
    """
    Test importing an already imported module returns the existing module.

    This test verifies that when attempting to import a module that's already
    in sys.modules, the function returns the existing module instead of
    reloading it.

    Setup:
        - Creates a mock module
        - Adds mock module to sys.modules
    """
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"already_imported_module": mock_module}):
        import_module_from_path("already_imported_module", "some_path")
        assert sys.modules["already_imported_module"] is mock_module


def test_import_module_from_path_with_init():
    """
    Test successful module import from a valid path with __init__.py.

    This test verifies the complete module import process:
    1. Path resolution and validation
    2. Module spec creation
    3. Module creation from spec
    4. Module execution
    5. Module registration in sys.modules

    Setup:
        - Patches Path for file validation
        - Patches spec creation and module creation utilities
        - Creates mock spec and module objects

    The test verifies all steps in the import process are called correctly
    and the module is properly registered in sys.modules.
    """
    mock_spec = MagicMock()
    mock_module = MagicMock()

    with (
        patch("lighter.utils.dynamic_imports.Path") as mock_path,
        patch("lighter.utils.dynamic_imports.importlib.util.spec_from_file_location") as mock_spec_from_file,
        patch("lighter.utils.dynamic_imports.importlib.util.module_from_spec") as mock_module_from_spec,
    ):
        # Setup mocks
        mock_path.return_value.resolve.return_value.__truediv__.return_value.is_file.return_value = True
        mock_spec_from_file.return_value = mock_spec
        mock_module_from_spec.return_value = mock_module

        # Execute function
        import_module_from_path("valid_module", "valid_path")

        # Verify mock interactions
        mock_path.assert_called_once()
        mock_spec_from_file.assert_called_once()
        mock_module_from_spec.assert_called_once_with(mock_spec)
        mock_spec.loader.exec_module.assert_called_once_with(mock_module)
        assert sys.modules["valid_module"] is mock_module
