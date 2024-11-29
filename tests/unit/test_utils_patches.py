from torch.nn import Linear, Module
import pytest

from lighter.utils.patches import PatchedModuleDict

def test_patched_module_dict_handles_reserved_names():
    # Test with previously problematic reserved names
    reserved_names = {
        'type': None,
        'to': None,
        'forward': 1,
        'training': "123",
        'modules': [Linear(10, 10)]
    }
    
    # Should work without raising exceptions
    patched_dict = PatchedModuleDict(reserved_names)
    
    # Verify all keys are accessible
    for key in reserved_names:
        assert key in patched_dict
        assert isinstance(patched_dict[key], Linear)
    
    # Test dictionary operations
    assert set(patched_dict.keys()) == set(reserved_names.keys())
    assert len(patched_dict.values()) == len(reserved_names)
    
    # Test deletion
    del patched_dict['type']
    assert 'type' not in patched_dict
