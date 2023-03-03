import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

from lighter.utils.dynamic_imports import import_module_from_path


def test_import_module_from_path(mocker):
    # Test case for importing a valid module.
    module_name = "test_module"
    module_path = "/path/to/test_module"
    module_file = f"{module_path}/__init__.py"
    mocker.patch.object(Path, "resolve", return_value=Path(module_path))
    mocker.patch.object(Path, "is_file", return_value=True)
    mock_spec = mocker.Mock()
    mocker.patch.object(importlib.util, "spec_from_file_location", return_value=mock_spec)
    mock_module = mocker.Mock()
    mock_spec.loader.exec_module = mocker.Mock(return_value=mock_module)

    import_module_from_path(module_name, module_path)

    mock_spec.loader.exec_module.assert_called_once_with(mock_module)
    assert module_name in sys.modules
    assert sys.modules[module_name] == mock_module

    # Test case for non-existing module file.
    mocker.patch.object(Path, "is_file", return_value=False)

    with pytest.raises(FileNotFoundError):
        import_module_from_path(module_name, "/path/to/non_existing_module")

    # Test case for importing a module that raises an exception during execution.
    mocker.patch.object(Path, "is_file", return_value=True)
    mock_spec.loader.exec_module.side_effect = Exception("Test exception")

    with pytest.raises(Exception, match="Test exception"):
        import_module_from_path(module_name, module_path)
