import pytest
from lighter.utils.patches import PatchedModuleDict
from torch.nn import Linear

def test_patched_module_dict():
    modules = {"layer1": Linear(10, 10)}
    patched_dict = PatchedModuleDict(modules)
    assert "layer1" in patched_dict
