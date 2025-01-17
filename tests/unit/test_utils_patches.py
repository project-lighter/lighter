import pytest
from torch.nn import Linear

from lighter.utils.patches import PatchedModuleDict


def test_patched_module_dict_basic_operations():
    """
    Test basic operations of PatchedModuleDict including initialization,
    containment checks, and deletion.
    """
    # Test initialization with modules and basic operations
    modules = {"layer1": Linear(10, 10), "layer2": Linear(10, 10)}
    patched_dict = PatchedModuleDict(modules)

    # Test initialization and __contains__
    assert len(patched_dict) == 2
    assert "layer1" in patched_dict
    assert "layer2" in patched_dict
    assert "non_existent" not in patched_dict

    # Test deletion
    del patched_dict["layer1"]
    assert "layer1" not in patched_dict
    assert len(patched_dict) == 1


def test_patched_module_dict_non_existent_key():
    """
    Test that accessing a non-existent key in PatchedModuleDict raises KeyError.
    """
    patched_dict = PatchedModuleDict()
    with pytest.raises(KeyError):
        _ = patched_dict["non_existent"]


def test_patched_module_dict_collection_methods():
    """
    Test collection methods of PatchedModuleDict including keys(), values(),
    and items().
    """
    modules = {"key1": Linear(10, 10), "key2": Linear(10, 10)}
    patched_dict = PatchedModuleDict(modules)

    # Test keys
    keys = patched_dict.keys()
    assert "key1" in keys
    assert "key2" in keys

    # Test values
    values = list(patched_dict.values())
    assert len(values) == 2
    assert all(isinstance(value, Linear) for value in values)

    # Test items
    items = dict(patched_dict.items())
    assert len(items) == 2
    assert all(isinstance(value, Linear) for value in items.values())
    assert set(items.keys()) == {"key1", "key2"}


def test_patched_module_dict_key_conflicts():
    """
    Test handling of key assignments and potential conflicts in PatchedModuleDict.
    """
    patched_dict = PatchedModuleDict()

    # Test multiple key assignments
    module1 = Linear(10, 10)
    patched_dict["_key"] = module1
    patched_dict["__key"] = Linear(20, 20)

    module3 = Linear(30, 30)
    patched_dict["key"] = module3

    assert patched_dict["key"] is module3
