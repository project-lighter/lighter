from torch.nn import Linear, Module

from lighter.utils.patches import PatchedModuleDict


def test_patched_module_dict_handles_reserved_names():
    # Test with previously problematic reserved names
    reserved_names = {
        "type": None,
        "to": None,
        "forward": Linear(10, 10),
        "training": Linear(10, 10),
    }

    # Should work without raising exceptions
    patched_dict = PatchedModuleDict(reserved_names)

    # Verify all keys are accessible
    for key in reserved_names:
        assert key in patched_dict
        assert patched_dict[key] == reserved_names[key]

    # Test deletion
    del patched_dict["type"]
    assert "type" not in patched_dict
