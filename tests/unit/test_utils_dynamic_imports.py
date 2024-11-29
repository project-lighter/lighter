import pytest

from lighter.utils.dynamic_imports import import_module_from_path


def test_import_module_from_path():
    with pytest.raises(FileNotFoundError):
        import_module_from_path("non_existent_module", "non_existent_path")
