import pytest
from torch.nn import Linear, Module

from lighter.utils.patches import PatchedModuleDict


def test_patched_module_dict_basic_operations():
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
    patched_dict = PatchedModuleDict()
    with pytest.raises(KeyError):
        _ = patched_dict["non_existent"]


def test_patched_module_dict_collection_methods():
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
    patched_dict = PatchedModuleDict()

    # Test multiple internal key iterations
    module1 = Linear(10, 10)
    patched_dict._modules["_key"] = module1
    patched_dict._modules["__key"] = Linear(20, 20)

    module3 = Linear(30, 30)
    patched_dict["key"] = module3

    assert patched_dict["key"] is module3
