import pytest
import torch
from torch.nn import Identity, Linear, Sequential

from lighter.utils.model import (
    adjust_prefix_and_load_state_dict,
    remove_n_last_layers_sequentially,
    replace_layer_with,
    replace_layer_with_identity,
)


@pytest.fixture
def dummy_model():
    """
    Creates a simple PyTorch model with two linear layers for testing purposes.

    Returns:
        torch.nn.Module: A model with two linear layers (10->10 dimensions each).
    """

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = Linear(10, 10)
            self.layer2 = Linear(10, 10)

        def forward(self, x):
            return self.layer2(self.layer1(x))

    return DummyModel()


@pytest.fixture
def sequential_model():
    """
    Creates a sequential PyTorch model with three linear layers.

    Returns:
        torch.nn.Sequential: A sequential model with three 10->10 dimensional linear layers.
    """
    return Sequential(Linear(10, 10), Linear(10, 10), Linear(10, 10))


@pytest.fixture
def state_dict_file(tmp_path):
    """
    Creates a temporary file containing a state dict for a simple model.

    Args:
        tmp_path: Pytest fixture providing a temporary directory unique to each test.

    Returns:
        tuple: (Path to saved state dict, The state dict dictionary)
    """
    model = torch.nn.Module()
    model.layer1 = Linear(10, 10)
    model.layer2 = Linear(10, 10)

    state_dict = {
        "layer1.weight": torch.randn(10, 10),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(10, 10),
        "layer2.bias": torch.randn(10),
    }

    path = tmp_path / "test_ckpt.pth"
    torch.save(state_dict, path)  # nosec B614
    return path, state_dict


@pytest.fixture
def state_dict_with_prefix_file(tmp_path):
    """
    Creates a temporary file containing a state dict with 'model.' prefix.

    Args:
        tmp_path: Pytest fixture providing a temporary directory unique to each test.

    Returns:
        tuple: (Path to saved state dict, The state dict dictionary with prefixes)
    """
    state_dict_with_prefix = {
        "model.layer1.weight": torch.randn(10, 10),
        "model.layer1.bias": torch.randn(10),
        "model.layer2.weight": torch.randn(10, 10),
        "model.layer2.bias": torch.randn(10),
    }

    path = tmp_path / "test_ckpt_with_prefix.pth"
    torch.save({"state_dict": state_dict_with_prefix}, path)  # nosec B614
    return path, state_dict_with_prefix


@pytest.fixture
def state_dict_with_custom_prefix_file(tmp_path):
    """
    Creates a temporary file containing a state dict with custom prefixes.

    Args:
        tmp_path: Pytest fixture providing a temporary directory unique to each test.

    Returns:
        tuple: (Path to saved state dict, The state dict dictionary with custom prefixes)
    """
    state_dict_with_prefix = {
        "prefix1.layer1.weight": torch.randn(10, 10),
        "prefix1.layer1.bias": torch.randn(10),
        "prefix2.layer2.weight": torch.randn(10, 10),
        "prefix2.layer2.bias": torch.randn(10),
    }

    path = tmp_path / "test_ckpt_with_prefix_adjustment.pth"
    torch.save(state_dict_with_prefix, path)  # nosec B614
    return path, state_dict_with_prefix


def test_replace_layer_with(dummy_model):
    """
    Tests if a layer in the model can be successfully replaced with a new layer.

    Args:
        dummy_model: Fixture providing a simple model for testing.
    """
    new_layer = Linear(10, 4)
    replace_layer_with(dummy_model, "layer1", new_layer)
    assert dummy_model.layer1 == new_layer


def test_replace_layer_with_identity(dummy_model):
    """
    Tests if a layer can be successfully replaced with an Identity layer.

    Args:
        dummy_model: Fixture providing a simple model for testing.
    """
    replace_layer_with_identity(dummy_model, "layer1")
    assert isinstance(dummy_model.layer1, Identity)


def test_remove_n_last_layers_sequentially(sequential_model):
    """
    Tests removing the last n layers from a sequential model.

    Args:
        sequential_model: Fixture providing a sequential model with three layers.

    Tests:
        - Removing one layer reduces model length by 1
        - Removing two layers reduces model length by 2
    """
    new_model = remove_n_last_layers_sequentially(sequential_model, num_layers=1)
    assert len(new_model) == 2

    new_model = remove_n_last_layers_sequentially(sequential_model, num_layers=2)
    assert len(new_model) == 1


def test_adjust_prefix_and_load_state_dict_basic(dummy_model, state_dict_file):
    """
    Tests basic functionality of loading a state dict into a model.

    Args:
        dummy_model: Fixture providing a simple model for testing.
        state_dict_file: Fixture providing a state dict file and its contents.

    Tests:
        - Weights change after loading
        - New weights match the state dict values
    """
    path, state_dict = state_dict_file

    # Store initial random weights
    initial_weights = {
        "layer1.weight": dummy_model.layer1.weight.clone(),
        "layer1.bias": dummy_model.layer1.bias.clone(),
        "layer2.weight": dummy_model.layer2.weight.clone(),
        "layer2.bias": dummy_model.layer2.bias.clone(),
    }

    # Load state dict
    adjust_prefix_and_load_state_dict(dummy_model, path)

    # Verify weights changed and match state dict
    assert not torch.equal(dummy_model.layer1.weight, initial_weights["layer1.weight"])
    assert not torch.equal(dummy_model.layer1.bias, initial_weights["layer1.bias"])
    assert torch.equal(dummy_model.layer1.weight, state_dict["layer1.weight"])
    assert torch.equal(dummy_model.layer1.bias, state_dict["layer1.bias"])


def test_adjust_prefix_and_load_state_dict_with_ignored_layers(dummy_model, state_dict_file):
    """
    Tests loading a state dict while ignoring specific layers.

    Args:
        dummy_model: Fixture providing a simple model for testing.
        state_dict_file: Fixture providing a state dict file and its contents.

    Tests:
        - Ignored layers maintain their original values
    """
    path, _ = state_dict_file

    # Store original values of layer2
    original_weight = dummy_model.layer2.weight.clone()
    original_bias = dummy_model.layer2.bias.clone()

    # Load state dict with ignored layers
    adjust_prefix_and_load_state_dict(dummy_model, path, layers_to_ignore=["layer2.weight", "layer2.bias"])

    # Verify ignored layers remain unchanged
    assert torch.equal(dummy_model.layer2.weight, original_weight)
    assert torch.equal(dummy_model.layer2.bias, original_bias)


def test_adjust_prefix_and_load_state_dict_with_model_prefix(dummy_model, state_dict_with_prefix_file):
    """
    Tests loading a state dict that has 'model.' prefix in its keys.

    Args:
        dummy_model: Fixture providing a simple model for testing.
        state_dict_with_prefix_file: Fixture providing a state dict file with 'model.' prefixed keys.

    Tests:
        - Model loads correctly despite prefix differences
        - Weights match the state dict values after prefix adjustment
    """
    path, state_dict = state_dict_with_prefix_file

    # Load state dict
    adjust_prefix_and_load_state_dict(dummy_model, path)

    # Verify weights match state dict (without the "model." prefix)
    assert torch.equal(dummy_model.layer1.weight, state_dict["model.layer1.weight"])
    assert torch.equal(dummy_model.layer1.bias, state_dict["model.layer1.bias"])


def test_adjust_prefix_and_load_state_dict_with_custom_prefix(dummy_model, state_dict_with_custom_prefix_file):
    """
    Tests loading a state dict with custom prefixes using prefix mapping.

    Args:
        dummy_model: Fixture providing a simple model for testing.
        state_dict_with_custom_prefix_file: Fixture providing a state dict file with custom prefixed keys.

    Tests:
        - Model loads correctly with custom prefix mapping
        - Weights match the state dict values after prefix adjustment
    """
    path, state_dict = state_dict_with_custom_prefix_file

    # Load state dict with prefix adjustments
    ckpt_to_model_prefix = {"prefix1": "", "prefix2": ""}
    adjust_prefix_and_load_state_dict(dummy_model, path, ckpt_to_model_prefix=ckpt_to_model_prefix)

    # Verify weights match state dict
    assert torch.equal(dummy_model.layer1.weight, state_dict["prefix1.layer1.weight"])
    assert torch.equal(dummy_model.layer1.bias, state_dict["prefix1.layer1.bias"])
    assert torch.equal(dummy_model.layer2.weight, state_dict["prefix2.layer2.weight"])
    assert torch.equal(dummy_model.layer2.bias, state_dict["prefix2.layer2.bias"])
