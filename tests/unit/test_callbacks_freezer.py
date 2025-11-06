"""Unit tests for the Freezer callback."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from lighter.callbacks.freezer import Freezer
from lighter.system import System


class DummyModel(nn.Module):
    """A simple model for testing Freezer."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 2)
        self.embedding = nn.Embedding(10, 10)


@pytest.fixture
def dummy_system():
    """Provides a mock System instance with a DummyModel."""
    system = MagicMock(spec=System)
    system.model = DummyModel()
    system.trainer = MagicMock()
    system.trainer.global_step = 0
    system.trainer.current_epoch = 0
    return system


def get_requires_grad_state(model):
    """Helper to get the requires_grad state of all parameters."""
    return {name: param.requires_grad for name, param in model.named_parameters()}


def test_freezer_init_validation():
    """Test Freezer initialization raises ValueError for invalid arguments."""
    with pytest.raises(ValueError, match="At least one of `names` or `name_starts_with` must be specified."):
        Freezer()
    with pytest.raises(ValueError, match="Only one of `until_step` or `until_epoch` can be specified."):
        Freezer(names="layer1.weight", until_step=1, until_epoch=1)


def test_freezer_freeze_by_name(dummy_system):
    """Test freezing parameters by exact name."""
    freezer = Freezer(names="layer1.weight")
    freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)

    state = get_requires_grad_state(dummy_system.model)
    assert not state["layer1.weight"]
    assert state["layer1.bias"]
    assert state["layer2.weight"]


def test_freezer_freeze_by_name_starts_with(dummy_system):
    """Test freezing parameters by name prefix."""
    freezer = Freezer(name_starts_with="layer1")
    freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)

    state = get_requires_grad_state(dummy_system.model)
    assert not state["layer1.weight"]
    assert not state["layer1.bias"]
    assert state["layer2.weight"]


def test_freezer_freeze_with_exceptions(dummy_system):
    """Test freezing with exceptions for specific parameters."""
    freezer = Freezer(name_starts_with="layer1", except_names="layer1.bias")
    freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)

    state = get_requires_grad_state(dummy_system.model)
    assert not state["layer1.weight"]
    assert state["layer1.bias"]
    assert state["layer2.weight"]


def test_freezer_freeze_with_except_name_starts_with(dummy_system):
    """Test freezing with exceptions for name prefixes."""
    freezer = Freezer(name_starts_with="layer", except_name_starts_with="layer2")
    freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)

    state = get_requires_grad_state(dummy_system.model)
    assert not state["layer1.weight"]
    assert not state["layer1.bias"]
    assert state["layer2.weight"]
    assert state["layer2.bias"]
    assert state["embedding.weight"]


def test_freezer_unfreeze_until_step(dummy_system):
    """Test unfreezing parameters after a specified step."""
    freezer = Freezer(names="layer1.weight", until_step=2)

    # Step 0: Should be frozen
    freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)
    assert not get_requires_grad_state(dummy_system.model)["layer1.weight"]

    # Step 1: Should still be frozen
    dummy_system.trainer.global_step = 1
    freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)
    assert not get_requires_grad_state(dummy_system.model)["layer1.weight"]

    # Step 2: Should be unfrozen
    dummy_system.trainer.global_step = 2
    freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)
    assert get_requires_grad_state(dummy_system.model)["layer1.weight"]

    # Step 3: Should remain unfrozen
    dummy_system.trainer.global_step = 3
    freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)
    assert get_requires_grad_state(dummy_system.model)["layer1.weight"]


def test_freezer_unfreeze_until_epoch(dummy_system):
    """Test unfreezing parameters after a specified epoch."""
    freezer = Freezer(names="layer1.weight", until_epoch=1)

    # Epoch 0, Step 0: Should be frozen
    freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)
    assert not get_requires_grad_state(dummy_system.model)["layer1.weight"]

    # Epoch 0, Step X: Still frozen
    dummy_system.trainer.global_step = 10
    freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)
    assert not get_requires_grad_state(dummy_system.model)["layer1.weight"]

    # Epoch 1, Step 0: Should be unfrozen
    dummy_system.trainer.current_epoch = 1
    dummy_system.trainer.global_step = 0  # Reset step for new epoch
    freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)
    assert get_requires_grad_state(dummy_system.model)["layer1.weight"]


def test_freezer_initial_state_unfrozen(dummy_system):
    """Test that parameters are initially unfrozen if until_step/epoch is 0."""
    freezer = Freezer(names="layer1.weight", until_step=0)

    # At step 0, it should immediately unfreeze
    freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)
    assert get_requires_grad_state(dummy_system.model)["layer1.weight"]


def test_freezer_multiple_calls_no_change(dummy_system):
    """Test that calling on_train_batch_start multiple times without state change does nothing."""
    freezer = Freezer(names="layer1.weight")

    with patch("lighter.callbacks.freezer.logger.info") as mock_logger_info:
        # First call, freezes
        freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)
        assert not get_requires_grad_state(dummy_system.model)["layer1.weight"]
        mock_logger_info.assert_called_once_with("Freezing the model.")
        mock_logger_info.reset_mock()

        # Second call, no change, should not log again
        freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)
        mock_logger_info.assert_not_called()

        # Unfreeze
        dummy_system.trainer.global_step = 10  # Beyond any until_step/epoch
        freezer.until_step = 5
        freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)
        assert get_requires_grad_state(dummy_system.model)["layer1.weight"]
        mock_logger_info.assert_called_once_with("Unfreezing the model.")
        mock_logger_info.reset_mock()

        # Call again, no change, should not log again
        freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)
        mock_logger_info.assert_not_called()


def test_freezer_with_system_model_attribute(dummy_system):
    """Test that Freezer correctly accesses the model attribute of a System instance."""
    freezer = Freezer(names="layer1.weight")
    freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)
    assert not get_requires_grad_state(dummy_system.model)["layer1.weight"]


def test_freezer_with_multiple_names_and_prefixes(dummy_system):
    """Test freezing with a combination of exact names and prefixes."""
    freezer = Freezer(names=["layer1.weight", "embedding.weight"], name_starts_with="layer2")
    freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)

    state = get_requires_grad_state(dummy_system.model)
    assert not state["layer1.weight"]
    assert state["layer1.bias"]
    assert not state["layer2.weight"]
    assert not state["layer2.bias"]
    assert not state["embedding.weight"]


def test_freezer_with_multiple_exceptions(dummy_system):
    """Test freezing with multiple exceptions (names and prefixes)."""
    freezer = Freezer(
        name_starts_with="layer",
        except_names=["layer1.bias"],
        except_name_starts_with=["layer2.weight"],
    )
    freezer.on_train_batch_start(dummy_system.trainer, dummy_system, None, 0)

    state = get_requires_grad_state(dummy_system.model)
    assert not state["layer1.weight"]
    assert state["layer1.bias"]
    assert state["layer2.weight"]
    assert not state["layer2.bias"]
    assert state["embedding.weight"]
