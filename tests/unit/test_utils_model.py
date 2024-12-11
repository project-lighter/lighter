import os

import pytest
import torch
from torch.nn import Linear, Module, Sequential

from lighter.utils.model import (
    adjust_prefix_and_load_state_dict,
    remove_n_last_layers_sequentially,
    replace_layer_with,
    replace_layer_with_identity,
)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(10, 10)
        self.layer2 = Linear(10, 10)

    def forward(self, x):
        return self.layer2(self.layer1(x))


def test_replace_layer_with():
    model = DummyModel()
    new_layer = Linear(10, 4)
    replace_layer_with(model, "layer1", new_layer)
    assert model.layer1 == new_layer


def test_replace_layer_with_identity():
    model = DummyModel()
    replace_layer_with_identity(model, "layer1")
    assert isinstance(model.layer1, torch.nn.Identity)


def test_remove_n_last_layers_sequentially():
    model = Sequential(Linear(10, 10), Linear(10, 10), Linear(10, 10))
    new_model = remove_n_last_layers_sequentially(model, num_layers=1)
    assert len(new_model) == 2

    model = Sequential(Linear(10, 10), Linear(10, 10), Linear(10, 10))
    new_model = remove_n_last_layers_sequentially(model, num_layers=2)
    assert len(new_model) == 1


def test_adjust_prefix_and_load_state_dict():
    class SimpleModel(Module):
        def __init__(self):
            super().__init__()
            self.layer1 = Linear(10, 10)
            self.layer2 = Linear(10, 10)

    model = SimpleModel()
    state_dict = {
        "layer1.weight": torch.randn(10, 10),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(10, 10),
        "layer2.bias": torch.randn(10),
    }
    torch.save(state_dict, "test_ckpt.pth")  # nosec B614

    # Test loading with no prefix adjustment
    loaded_model = adjust_prefix_and_load_state_dict(model, "test_ckpt.pth")
    assert torch.equal(loaded_model.layer1.weight, state_dict["layer1.weight"])
    assert torch.equal(loaded_model.layer1.bias, state_dict["layer1.bias"])

    # Test loading with layers to ignore
    loaded_model = adjust_prefix_and_load_state_dict(model, "test_ckpt.pth", layers_to_ignore=["layer2.weight", "layer2.bias"])
    assert torch.equal(loaded_model.layer2.weight, state_dict["layer2.weight"])
    assert torch.equal(loaded_model.layer2.bias, state_dict["layer2.bias"])

    # Test loading with "state_dict" key and "model." prefix
    state_dict_with_prefix = {
        "model.layer1.weight": torch.randn(10, 10),
        "model.layer1.bias": torch.randn(10),
        "model.layer2.weight": torch.randn(10, 10),
        "model.layer2.bias": torch.randn(10),
    }
    torch.save({"state_dict": state_dict_with_prefix}, "test_ckpt_with_prefix.pth")  # nosec B614

    loaded_model = adjust_prefix_and_load_state_dict(model, "test_ckpt_with_prefix.pth")
    assert torch.equal(loaded_model.layer1.weight, state_dict_with_prefix["model.layer1.weight"])
    assert torch.equal(loaded_model.layer1.bias, state_dict_with_prefix["model.layer1.bias"])
    assert torch.equal(loaded_model.layer2.weight, state_dict_with_prefix["model.layer2.weight"])
    assert torch.equal(loaded_model.layer2.bias, state_dict_with_prefix["model.layer2.bias"])

    # Test loading with prefix adjustments
    state_dict_with_prefix = {
        "prefix1.layer1.weight": torch.randn(10, 10),
        "prefix1.layer1.bias": torch.randn(10),
        "prefix2.layer2.weight": torch.randn(10, 10),
        "prefix2.layer2.bias": torch.randn(10),
    }
    torch.save(state_dict_with_prefix, "test_ckpt_with_prefix_adjustment.pth")  # nosec B614

    ckpt_to_model_prefix = {"prefix1": "", "prefix2": ""}

    loaded_model = adjust_prefix_and_load_state_dict(
        model, "test_ckpt_with_prefix_adjustment.pth", ckpt_to_model_prefix=ckpt_to_model_prefix
    )
    assert torch.equal(loaded_model.layer1.weight, state_dict_with_prefix["prefix1.layer1.weight"])
    assert torch.equal(loaded_model.layer1.bias, state_dict_with_prefix["prefix1.layer1.bias"])
    assert torch.equal(loaded_model.layer2.weight, state_dict_with_prefix["prefix2.layer2.weight"])
    assert torch.equal(loaded_model.layer2.bias, state_dict_with_prefix["prefix2.layer2.bias"])

    # Clean up
    os.remove("test_ckpt.pth")
    os.remove("test_ckpt_with_prefix.pth")
    os.remove("test_ckpt_with_prefix_adjustment.pth")
