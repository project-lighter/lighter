"""Unit tests for the PatchedModuleDict class in lighter/utils/patches.py"""

import pytest
from torch import nn

from lighter.utils.patches import PatchedModuleDict


@pytest.fixture
def sample_modules():
    """Returns a dictionary of sample modules."""
    return {
        "linear": nn.Linear(10, 5),
        "conv": nn.Conv2d(3, 8, 3),
    }


def test_patched_module_dict_init(sample_modules):
    """Test initialization of PatchedModuleDict."""
    module_dict = PatchedModuleDict(sample_modules)
    assert "linear" in module_dict
    assert "conv" in module_dict
    assert module_dict["linear"] == sample_modules["linear"]


def test_patched_module_dict_setitem_getitem(sample_modules):
    """Test __setitem__ and __getitem__ methods."""
    module_dict = PatchedModuleDict()
    module_dict["linear"] = sample_modules["linear"]
    module_dict["conv"] = sample_modules["conv"]
    assert module_dict["linear"] == sample_modules["linear"]
    assert module_dict["conv"] == sample_modules["conv"]


def test_patched_module_dict_key_conflict():
    """Test that key conflicts are handled."""
    module_dict = PatchedModuleDict()
    module1 = nn.Linear(1, 1)
    module2 = nn.Linear(1, 1)
    # In a normal ModuleDict, this would be problematic if keys were mangled to the same internal key.
    # The patch is exactly for this.
    module_dict["_key"] = module1
    module_dict["key"] = module2
    assert module_dict["_key"] is module1
    assert module_dict["key"] is module2


def test_patched_module_dict_delitem(sample_modules):
    """Test __delitem__ method."""
    module_dict = PatchedModuleDict(sample_modules)
    del module_dict["linear"]
    assert "linear" not in module_dict
    assert "conv" in module_dict


def test_patched_module_dict_contains(sample_modules):
    """Test __contains__ method."""
    module_dict = PatchedModuleDict(sample_modules)
    assert "linear" in module_dict
    assert "non_existent" not in module_dict


def test_patched_module_dict_get(sample_modules):
    """Test get method."""
    module_dict = PatchedModuleDict(sample_modules)
    assert module_dict.get("linear") == sample_modules["linear"]
    assert module_dict.get("non_existent") is None
    assert module_dict.get("non_existent", "default") == "default"


def test_patched_module_dict_keys(sample_modules):
    """Test keys method."""
    module_dict = PatchedModuleDict(sample_modules)
    keys = list(module_dict.keys())
    assert "linear" in keys
    assert "conv" in keys
    assert len(keys) == 2


def test_patched_module_dict_items(sample_modules):
    """Test items method."""
    module_dict = PatchedModuleDict(sample_modules)
    items = list(module_dict.items())
    assert ("linear", sample_modules["linear"]) in items
    assert ("conv", sample_modules["conv"]) in items
    assert len(items) == 2


def test_patched_module_dict_values(sample_modules):
    """Test values method."""
    module_dict = PatchedModuleDict(sample_modules)
    values = list(module_dict.values())
    assert sample_modules["linear"] in values
    assert sample_modules["conv"] in values
    assert len(values) == 2
