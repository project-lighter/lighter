import sys
from unittest.mock import MagicMock, patch

import pytest

from lighter.utils.dynamic_imports import import_module_from_path


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
        result = import_module_from_path("already_imported_module", "some_path")
        assert result is mock_module
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
    6. Module registration with cloudpickle for pickle-by-value serialization

    Setup:
        - Patches Path for file validation
        - Patches spec creation and module creation utilities
        - Patches cloudpickle.register_pickle_by_value
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
        patch("lighter.utils.dynamic_imports.cloudpickle.register_pickle_by_value") as mock_register,
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
        mock_register.assert_called_once_with(mock_module)
        assert sys.modules["valid_module"] is mock_module
