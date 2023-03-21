import sys
from pathlib import Path

import pytest

from lighter.utils.dynamic_imports import import_module_from_path


def test_import_module_from_path(tmpdir):
    """Test the import_module_from_path function.

    This test creates a temporary directory and file using the tmpdir fixture provided
    by pytest. It then tests that the import_module_from_path function successfully imports
    the module and adds it to sys.modules. It also tests that a FileNotFoundError is raised
    when trying to import from a non-existent directory.

    Args:
        tmpdir: A temporary directory path object provided by pytest.
    """
    # Create a temporary directory and file for testing
    temp_dir = Path(tmpdir)
    temp_file = temp_dir / "__init__.py"
    # Write some content to the temporary file
    temp_file.write_text("x = 1")

    # Test successful import
    # Call the function with the temporary directory path and a module name
    import_module_from_path("test_module", str(temp_dir))
    # Check that the module was successfully imported and added to sys.modules
    assert "test_module" in sys.modules

    # Test that the imported module can be used
    import test_module  # pylint: disable=import-outside-toplevel, import-error

    assert test_module.x == 1

    # Test FileNotFoundError
    # Call the function with a non-existent directory path
    with pytest.raises(FileNotFoundError):
        import_module_from_path("test_module", str(temp_dir / "non_existent_dir"))
