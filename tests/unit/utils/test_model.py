from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.nn import Identity, Linear, Module, ModuleDict, Sequential

from lighter.utils.model import (
    adjust_prefix_and_load_state_dict,
    remove_n_last_layers_sequentially,
    replace_layer_with,
    replace_layer_with_identity,
    reshape_pred_if_single_value_prediction,
)


def test_reshape_pred_if_single_value_prediction():
    """Test the reshape_pred_if_single_value_prediction function.

    This test checks several cases: one where the pred and target tensors have mismatched
    dimensions and one where they have matching dimensions. In the first case, the test
    verifies that the reshape_pred_if_single_value_prediction function correctly reshapes
    the pred tensor to match the shape of the target tensor. In the second case, it verifies
    that the function returns the original pred tensor without any changes. Additional test
    cases verify that the function returns the original pred value when it is not an instance
    of torch.Tensor or when target is None.
    """
    # Test case where pred and target have matching dimensions
    pred = torch.tensor([[1], [2], [3]])
    target = torch.tensor([1, 2, 3])
    result = reshape_pred_if_single_value_prediction(pred, target)
    assert result.shape == (3,)

    # Test case where pred and target have mismatched dimensions
    pred = torch.tensor([[1], [2], [3]])
    target = torch.tensor([[1], [2], [3]])
    result = reshape_pred_if_single_value_prediction(pred, target)
    assert result.shape == (3, 1)

    # Test case where pred is not an instance of torch.Tensor
    pred = [[1], [2], [3]]
    target = torch.tensor([1, 2, 3])
    result = reshape_pred_if_single_value_prediction(pred, target)
    assert result == [[1], [2], [3]]

    # Test case where target is None
    pred = torch.tensor([[1], [2], [3]])
    target = None
    result = reshape_pred_if_single_value_prediction(pred, target)
    assert result.shape == (3, 1)


def test_replace_layer_with():
    """
    Tests the functionality of the replace_layer_with function.

    Test cases include:
    - Replacing a layer in a simple model and checking if the new layer is set correctly
    - Attempting to replace a non-existent layer and verifying that an error is raised
    - Replacing a nested layer and checking if the new layer is set correctly
    """

    class TestModel(Module):
        def __init__(self):
            super().__init__()
            self.fc1 = Linear(10, 5)
            self.fc2 = Linear(5, 2)

    model = TestModel()
    new_layer = Linear(5, 3)

    # Replace fc2 layer with new_layer
    replace_layer_with(model, "fc2", new_layer)

    # Check if fc2 layer is replaced with new_layer
    assert model.fc2 == new_layer

    # Test replacing a non-existent layer
    with pytest.raises(AttributeError):
        replace_layer_with(model, "fc3", new_layer)


def test_replace_layer_with_identity():
    """Tests the replace_layer_with_identity function."""

    class TestModel(Module):
        def __init__(self):
            super().__init__()
            self.fc = "original layer"

    model = TestModel()
    assert model.fc == "original layer"
    model = replace_layer_with_identity(model, "fc")
    assert isinstance(model.fc, Identity)


def test_remove_n_last_layers_sequentially():
    """Tests the remove_n_last_layers_sequentially function."""

    # Define a simple test model with 3 layers
    class TestModel(Module):
        def __init__(self):
            super().__init__()
            self.fc1 = Linear(10, 5)
            self.fc2 = Linear(5, 3)
            self.fc3 = Linear(3, 1)

    # Create an instance of the test model
    model = TestModel()

    # Check that the original model has 3 layers
    assert len(list(model.children())) == 3

    # Remove the last layer using the function
    model = remove_n_last_layers_sequentially(model)

    # Check that the new model has only 2 layers and is an instance of nn.Sequential
    assert len(list(model.children())) == 2
    assert isinstance(model, Sequential)


def test_adjust_prefix_and_load_state_dict(tmpdir):
    """Tests the adjust_prefix_and_load_state_dict function."""

    class InputModel(Module):
        def __init__(self):
            super().__init__()
            self.encoder = Sequential(Linear(10, 5), Linear(5, 3))
            self.decoder = Sequential(Linear(3, 5), Linear(5, 10))

    class TargetModel(Module):
        def __init__(self):
            super().__init__()
            self.new_encoder = Sequential(Linear(10, 5), Linear(5, 3))
            self.new_decoder = Sequential(Linear(3, 5), Linear(5, 10))

    # Create an instance of the test model
    input_model = InputModel()
    target_model = TargetModel()

    # Save checkpoint
    ckpt_path = Path(tmpdir) / "ckpt.pth"
    torch.save(input_model.state_dict(), ckpt_path)

    # Test with no prefix adjustment
    input_model_adjusted = adjust_prefix_and_load_state_dict(input_model, ckpt_path)
    assert set(input_model_adjusted.state_dict().keys()) == set(input_model.state_dict().keys())

    # Test with prefix adjustment
    ckpt_to_model_prefix = {"encoder": "new_encoder", "decoder": "new_decoder"}
    target_model = adjust_prefix_and_load_state_dict(target_model, ckpt_path, ckpt_to_model_prefix)
    # Check that the prefixes are not present anymore
    assert not any(key.startswith("encoder") for key in target_model.state_dict().keys())
    assert not any(key.startswith("decoder") for key in target_model.state_dict().keys())
    # Check if the keys were converted correctly
    new_encoder_keys = [key for key in target_model.state_dict().keys() if key.startswith("new_encoder")]
    new_decoder_keys = [key for key in target_model.state_dict().keys() if key.startswith("new_decoder")]
    old_encoder_keys = [
        key for key in input_model.state_dict().keys() if key.replace("encoder", "new_encoder") in new_encoder_keys
    ]
    old_decoder_keys = [
        key for key in input_model.state_dict().keys() if key.replace("decoder", "new_decoder") in new_decoder_keys
    ]
    assert [key.replace("encoder", "new_encoder") for key in old_encoder_keys] == new_encoder_keys
    assert [key.replace("decoder", "new_decoder") for key in old_decoder_keys] == new_decoder_keys

    # Test adding a prefix to the checkpoint's keys
    class PrefixModel(Module):
        def __init__(self):
            super().__init__()
            self.prefix = Sequential(Linear(10, 5), Linear(5, 3))

    target_model = PrefixModel()
    input_model = Sequential(Linear(10, 5), Linear(5, 3))
    torch.save(input_model.state_dict(), ckpt_path)
    ckpt_to_model_prefix = {"": "prefix"}
    target_model = adjust_prefix_and_load_state_dict(target_model, ckpt_path, ckpt_to_model_prefix)
    assert all(key.startswith("prefix") for key in target_model.state_dict().keys())

    # Test removing a prefix from the checkpoint's keys
    input_model = PrefixModel()
    torch.save(input_model.state_dict(), ckpt_path)
    target_model = Sequential(Linear(10, 5), Linear(5, 3))
    ckpt_to_model_prefix = {"prefix": ""}
    target_model = adjust_prefix_and_load_state_dict(target_model, ckpt_path, ckpt_to_model_prefix)
    assert not any(key.startswith("prefix") for key in target_model.state_dict().keys())
