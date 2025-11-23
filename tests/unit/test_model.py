"""Unit tests for the redesigned LighterModule class."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchmetrics import Accuracy, MetricCollection

from lighter.model import LighterModule
from lighter.utils.types.enums import Mode

# Shared fixtures (mock_trainer, simple_model, dummy_dataset) are available
# from conftest.py and auto-discovered by pytest

# ============================================================================
# Helper Functions (imported from conftest for direct use)
# ============================================================================


def mock_trainer_state(trainer, training=False, validating=False, testing=False, predicting=False, sanity_checking=False):
    """Helper function to set trainer state flags for testing mode detection.

    Note: This is duplicated from conftest.py because it's used as a direct function
    call, not as a fixture.
    """
    trainer.training = training
    trainer.validating = validating
    trainer.testing = testing
    trainer.predicting = predicting
    trainer.sanity_checking = sanity_checking


# ============================================================================
# Helper Classes and Fixtures
# ============================================================================


class SimpleLighterModule(LighterModule):
    """Concrete implementation of LighterModule for testing."""

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.cross_entropy(pred, y)

        if self.train_metrics is not None:
            self.train_metrics(pred, y)

        return {"loss": loss, "pred": pred, "target": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.cross_entropy(pred, y)

        if self.val_metrics is not None:
            self.val_metrics(pred, y)

        return {"loss": loss, "pred": pred, "target": y}

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        if self.test_metrics is not None:
            self.test_metrics(pred, y)

        return {"pred": pred, "target": y}

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred


@pytest.fixture
def simple_system(simple_model):
    """Creates a SimpleLighterModule instance using the shared simple_model fixture."""
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(simple_model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    train_metrics = MetricCollection([Accuracy(task="multiclass", num_classes=2)])
    val_metrics = MetricCollection([Accuracy(task="multiclass", num_classes=2)])
    test_metrics = MetricCollection([Accuracy(task="multiclass", num_classes=2)])

    system = SimpleLighterModule(
        network=simple_model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )

    return system


# ============================================================================
# Test System Class
# ============================================================================


def test_system_requires_step_implementation(simple_model):
    """Test that LighterModule raises NotImplementedError when step methods are called."""
    system = LighterModule(network=simple_model)

    # System can be instantiated, but calling unimplemented steps should raise
    batch = (torch.randn(2, 4), torch.tensor([0, 1]))

    with pytest.raises(NotImplementedError, match="must implement training_step"):
        system.training_step(batch, batch_idx=0)

    with pytest.raises(NotImplementedError, match="must implement validation_step"):
        system.validation_step(batch, batch_idx=0)

    with pytest.raises(NotImplementedError, match="must implement test_step"):
        system.test_step(batch, batch_idx=0)

    with pytest.raises(NotImplementedError, match="must implement predict_step"):
        system.predict_step(batch, batch_idx=0)


def test_system_initialization(simple_system):
    """Check that attributes are correctly set after initialization."""
    assert isinstance(simple_system.network, nn.Module)
    assert simple_system.criterion is not None
    assert simple_system.optimizer is not None
    assert simple_system.scheduler is not None
    assert simple_system.train_metrics is not None
    assert simple_system.val_metrics is not None
    assert simple_system.test_metrics is not None


def test_prepare_metrics(simple_model):
    """Test _prepare_metrics handles different input types."""

    # Test with None
    system = SimpleLighterModule(network=simple_model)
    assert system.train_metrics is None

    # Test with single Metric
    metric = Accuracy(task="multiclass", num_classes=2)
    system = SimpleLighterModule(network=simple_model, train_metrics=metric)
    assert isinstance(system.train_metrics, MetricCollection)

    # Test with list of Metrics
    metrics = [Accuracy(task="multiclass", num_classes=2)]
    system = SimpleLighterModule(network=simple_model, train_metrics=metrics)
    assert isinstance(system.train_metrics, MetricCollection)

    # Test with MetricCollection
    metrics = MetricCollection([Accuracy(task="multiclass", num_classes=2)])
    system = SimpleLighterModule(network=simple_model, train_metrics=metrics)
    assert isinstance(system.train_metrics, MetricCollection)


def test_configure_optimizers(simple_system):
    """Test configure_optimizers returns correct structure."""
    opt_config = simple_system.configure_optimizers()
    assert isinstance(opt_config, dict)
    assert "optimizer" in opt_config
    assert "lr_scheduler" in opt_config


def test_configure_optimizers_without_optimizer(simple_model):
    """Test configure_optimizers when no optimizer is provided."""
    system = SimpleLighterModule(network=simple_model, optimizer=None)

    with pytest.raises(ValueError, match="Optimizer not configured"):
        system.configure_optimizers()


def test_configure_optimizers_without_scheduler(simple_model):
    """Test configure_optimizers when no scheduler is provided."""
    optimizer = SGD(simple_model.parameters(), lr=0.01)
    system = SimpleLighterModule(network=simple_model, optimizer=optimizer, scheduler=None)

    opt_config = system.configure_optimizers()
    assert isinstance(opt_config, dict)
    assert "optimizer" in opt_config
    assert "lr_scheduler" not in opt_config


def test_forward_delegates_to_model(simple_system):
    """Test that forward() delegates to self.network()."""
    x = torch.randn(2, 4)
    output = simple_system(x)

    # Should have same output as calling network directly
    expected = simple_system.network(x)
    assert torch.allclose(output, expected)


def test_mode_property(simple_system, mock_trainer):
    """Test mode property detects mode from trainer state."""
    simple_system.trainer = mock_trainer

    # Test training mode
    mock_trainer_state(mock_trainer, training=True)
    assert simple_system.mode == Mode.TRAIN

    # Test validation mode
    mock_trainer_state(mock_trainer, validating=True)
    assert simple_system.mode == Mode.VAL

    # Test test mode
    mock_trainer_state(mock_trainer, testing=True)
    assert simple_system.mode == Mode.TEST

    # Test predict mode
    mock_trainer_state(mock_trainer, predicting=True)
    assert simple_system.mode == Mode.PREDICT

    # Test sanity checking (should return VAL)
    mock_trainer_state(mock_trainer, sanity_checking=True)
    assert simple_system.mode == Mode.VAL


def test_mode_without_trainer(simple_system):
    """Test mode property raises error when no trainer attached."""
    simple_system.trainer = None
    with pytest.raises(RuntimeError, match="is not attached to a"):
        _ = simple_system.mode


def test_mode_undetermined_state(simple_system, mock_trainer):
    """Test mode property raises error when trainer is in undetermined state."""
    simple_system.trainer = mock_trainer
    # Set all flags to False (no active mode)
    mock_trainer_state(mock_trainer, training=False, validating=False, testing=False, predicting=False, sanity_checking=False)

    with pytest.raises(RuntimeError, match="Cannot determine mode"):
        _ = simple_system.mode


# Dataloader tests removed - dataloaders are now configured via data: key in config


def test_validate_and_log_simple_loss(simple_system, mock_trainer):
    """Test _log_outputs with simple scalar loss."""
    simple_system.trainer = mock_trainer
    mock_trainer.training = True
    simple_system.log = MagicMock()

    output = {"loss": torch.tensor(1.0)}
    simple_system._log_outputs(output, batch_idx=0)

    # Should log loss twice (step + epoch)
    assert simple_system.log.call_count >= 2


def test_validate_and_log_dict_loss(simple_system, mock_trainer):
    """Test _log_outputs with multi-component loss dict."""
    simple_system.trainer = mock_trainer
    mock_trainer.training = True
    simple_system.log = MagicMock()

    loss_dict = {"total": torch.tensor(3.0), "ce": torch.tensor(2.0), "reg": torch.tensor(1.0)}
    output = {"loss": loss_dict}
    simple_system._log_outputs(output, batch_idx=0)

    # Should log each loss component twice (step + epoch)
    # 3 components * 2 (step + epoch) = 6 calls minimum
    assert simple_system.log.call_count >= 6


def test_validate_and_log_none_loss(simple_system, mock_trainer):
    """Test _log_outputs handles None loss gracefully."""
    simple_system.trainer = mock_trainer
    mock_trainer.training = True
    simple_system.log = MagicMock()

    # Output with no loss key or loss=None
    output = {"loss": None}
    simple_system._log_outputs(output, batch_idx=0)

    # log should not be called for loss when it's None
    # (may be called for other things, but not for loss)
    # We're testing that it doesn't crash
    assert True  # If we reach here, test passes


def test_validate_and_log_dict_loss_without_total(simple_model, mock_trainer):
    """Test _on_batch_end raises error when loss dict missing 'total' key."""
    system = SimpleLighterModule(network=simple_model)
    system.trainer = mock_trainer
    mock_trainer.training = True

    loss_dict = {"ce": torch.tensor(2.0), "reg": torch.tensor(1.0)}  # Missing 'total'
    output = {"loss": loss_dict}

    with pytest.raises(ValueError, match="Loss dict.*must include 'total' key"):
        system._on_batch_end(output, batch_idx=0)


def test_validate_and_log_without_logger(simple_system, mock_trainer):
    """Test _log_outputs does nothing when no logger."""
    simple_system.trainer = mock_trainer
    mock_trainer.logger = None

    output = {"loss": torch.tensor(1.0)}
    # Should not raise any errors
    simple_system._log_outputs(output, batch_idx=0)


def test_batch_end_hooks_call_validate_and_log(simple_system, mock_trainer):
    """Test that batch-end hooks call _log_outputs."""
    simple_system.trainer = mock_trainer
    simple_system._log_outputs = MagicMock()

    output = {"loss": torch.tensor(1.0)}
    batch = (torch.randn(2, 4), torch.tensor([0, 1]))

    # Test on_train_batch_end
    simple_system.on_train_batch_end(output, batch, batch_idx=0)
    simple_system._log_outputs.assert_called_once_with(output, 0)

    # Test on_validation_batch_end
    simple_system._log_outputs.reset_mock()
    simple_system.on_validation_batch_end(output, batch, batch_idx=0)
    simple_system._log_outputs.assert_called_once_with(output, 0)

    # Test on_test_batch_end
    simple_system._log_outputs.reset_mock()
    simple_system.on_test_batch_end(output, batch, batch_idx=0)
    simple_system._log_outputs.assert_called_once_with(output, 0)


def test_predict_step_implementation(simple_system):
    """Test predict_step implementation."""
    batch = (torch.randn(2, 4), torch.tensor([0, 1]))
    output = simple_system.predict_step(batch, batch_idx=0)

    # Should return predictions
    x, y = batch
    expected = simple_system(x)
    assert torch.allclose(output, expected)


def test_training_step_returns_dict(simple_system):
    """Test that training_step returns dict."""
    simple_system.trainer = MagicMock()
    mock_trainer_state(simple_system.trainer, training=True)

    # Create batch manually
    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])
    batch = (x, y)

    output = simple_system.training_step(batch, batch_idx=0)

    assert isinstance(output, dict)
    assert "loss" in output
    assert "pred" in output
    assert "target" in output


def test_validation_step_returns_dict(simple_system):
    """Test that validation_step returns dict."""
    simple_system.trainer = MagicMock()
    mock_trainer_state(simple_system.trainer, validating=True)

    # Create batch manually
    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])
    batch = (x, y)

    output = simple_system.validation_step(batch, batch_idx=0)

    assert isinstance(output, dict)
    assert "loss" in output
    assert "pred" in output


def test_test_step_returns_dict(simple_system):
    """Test that test_step returns dict."""
    simple_system.trainer = MagicMock()
    mock_trainer_state(simple_system.trainer, testing=True)

    # Create batch manually
    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])
    batch = (x, y)

    output = simple_system.test_step(batch, batch_idx=0)

    assert isinstance(output, dict)
    assert "loss" not in output  # Test mode doesn't require loss
    assert "pred" in output


def test_metrics_logging_in_validate_and_log(simple_system):
    """Test that metrics are logged in _log_outputs."""
    simple_system.trainer = MagicMock()
    simple_system.trainer.logger = MagicMock()
    simple_system.trainer.training = True
    simple_system.log = MagicMock()

    # Call train_metrics manually to update them
    pred = torch.randn(2, 2)
    target = torch.tensor([0, 1])
    simple_system.train_metrics(pred, target)

    output = {"loss": torch.tensor(1.0), "pred": pred, "target": target}
    simple_system._log_outputs(output, batch_idx=0)

    # Check that log was called for metrics
    # Should have calls for loss (2) + metrics (2 per metric) + optimizer stats (2)
    assert simple_system.log.call_count >= 2


def test_optimizer_stats_logged_once_per_epoch(simple_system):
    """Test that optimizer stats are logged only once per epoch (batch_idx=0)."""
    from unittest.mock import patch

    simple_system.trainer = MagicMock()
    simple_system.trainer.training = True
    simple_system.trainer.validating = False
    simple_system.trainer.testing = False
    simple_system.trainer.predicting = False
    simple_system.trainer.sanity_checking = False

    # batch_idx=0 should call get_optimizer_stats
    with patch("lighter.model.get_optimizer_stats") as mock_get_stats:
        mock_get_stats.return_value = {"lr": 0.01}
        simple_system._log_optimizer_stats(batch_idx=0)
        assert mock_get_stats.called

    # batch_idx=1 should NOT call get_optimizer_stats
    with patch("lighter.model.get_optimizer_stats") as mock_get_stats:
        mock_get_stats.return_value = {"lr": 0.01}
        simple_system._log_optimizer_stats(batch_idx=1)
        assert not mock_get_stats.called


def test_log_method_sets_sync_dist_for_epoch(simple_system):
    """Test that _log sets sync_dist=True for epoch logging."""
    simple_system.trainer = MagicMock()
    mock_trainer_state(simple_system.trainer, training=True)
    simple_system.log = MagicMock()

    simple_system._log("test/metric", torch.tensor(1.0), on_epoch=True, sync_dist=True)

    # Check that epoch logging was called with sync_dist=True
    assert simple_system.log.call_count == 1
    call_kwargs = simple_system.log.call_args[1]
    assert call_kwargs["sync_dist"] is True
    assert call_kwargs["on_epoch"] is True
    assert call_kwargs["on_step"] is False


def test_normalize_output_accepts_dict(simple_model):
    """Test that _normalize_output accepts and passes through dict."""
    system = SimpleLighterModule(network=simple_model)

    output = {"loss": torch.tensor(1.0), "pred": torch.randn(2, 2)}
    normalized = system._normalize_output(output)
    assert normalized is output  # Should be the same dict


def test_normalize_output_accepts_tensor(simple_model):
    """Test that _normalize_output converts Tensor to dict."""
    system = SimpleLighterModule(network=simple_model)

    output = torch.tensor(1.0)
    normalized = system._normalize_output(output)
    assert isinstance(normalized, dict)
    assert "loss" in normalized
    assert torch.allclose(normalized["loss"], output)


def test_normalize_output_rejects_invalid_types(simple_model):
    """Test that _normalize_output rejects invalid types."""
    system = SimpleLighterModule(network=simple_model)

    # List should be rejected
    bad_output = [1.0, 2.0]
    with pytest.raises(TypeError, match="must return torch.Tensor or dict"):
        system._normalize_output(bad_output)

    # String should be rejected
    bad_output = "invalid"
    with pytest.raises(TypeError, match="must return torch.Tensor or dict"):
        system._normalize_output(bad_output)


def test_normalize_output_validates_loss_dict_has_total(simple_model):
    """Test that _normalize_output validates loss dict has 'total' key."""
    system = SimpleLighterModule(network=simple_model)

    # Loss dict with 'total' should be accepted
    valid_output = {"loss": {"total": torch.tensor(3.0), "ce": torch.tensor(2.0)}}
    normalized = system._normalize_output(valid_output)
    assert normalized is valid_output

    # Loss dict without 'total' should be rejected
    invalid_output = {"loss": {"ce": torch.tensor(2.0), "reg": torch.tensor(1.0)}}
    with pytest.raises(ValueError, match="Loss dict must include 'total' key"):
        system._normalize_output(invalid_output)

    # Loss as tensor should still be accepted
    tensor_loss_output = {"loss": torch.tensor(1.0), "pred": torch.randn(2, 2)}
    normalized = system._normalize_output(tensor_loss_output)
    assert normalized is tensor_loss_output


def test_batch_end_hooks_accept_tensor(simple_system, mock_trainer):
    """Test that batch-end hooks accept and normalize tensor outputs."""
    simple_system.trainer = mock_trainer
    mock_trainer.training = True
    simple_system.log = MagicMock()

    # Tensor should be accepted and normalized
    tensor_output = torch.tensor(1.0)
    batch = (torch.randn(2, 4), torch.tensor([0, 1]))

    # Should not raise
    simple_system.on_train_batch_end(tensor_output, batch, batch_idx=0)
    # Should have called log
    assert simple_system.log.call_count > 0
