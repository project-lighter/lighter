"""Unit tests for the System class."""

from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy

from lighter.system import System
from lighter.utils.types.enums import Data, Mode


class DummyDataset(Dataset):
    """Dataset returning (input_tensor, target_int)"""

    def __init__(self, size=8, with_target=True):
        super().__init__()
        self.size = size
        self.with_target = with_target
        self.data = []
        for _ in range(self.size):
            x = torch.randn(4)
            if self.with_target:
                y = torch.randint(0, 2, size=()).item()  # scalar int
                self.data.append((x, y))
            else:
                self.data.append(x)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.size


class SimpleModel(nn.Module):
    """Simple model with a single linear layer"""

    def __init__(self, in_features=4, out_features=2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, epoch=None, step=None):
        return self.linear(x)


@pytest.fixture
def dummy_dataloaders():
    """Provides train/val/test/predict DataLoaders"""
    return {
        "train": DataLoader(DummyDataset(size=8, with_target=True), batch_size=2),
        "val": DataLoader(DummyDataset(size=4, with_target=True), batch_size=2),
        "test": DataLoader(DummyDataset(size=4, with_target=True), batch_size=2),
        "predict": DataLoader(DummyDataset(size=4, with_target=False), batch_size=2),
    }


@pytest.fixture
def simple_system(dummy_dataloaders):
    """Creates a System instance with a mock trainer and mocked log method"""
    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    metrics = {
        "train": Accuracy(task="multiclass", num_classes=2),
        "val": Accuracy(task="multiclass", num_classes=2),
        "test": Accuracy(task="multiclass", num_classes=2),
    }

    system = System(
        model=model,
        dataloaders=dummy_dataloaders,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        metrics=metrics,
        adapters=None,
        inferer=None,
    )

    # Initialize a Trainer without logger and checkpointing
    trainer = pl.Trainer(logger=None, enable_checkpointing=False, max_epochs=1)
    system.trainer = trainer

    # Mock _log_stats to prevent actual logging
    system._log_stats = MagicMock()

    return system


def test_system_initialization(simple_system):
    """Check that attributes are correctly set after initialization."""
    assert isinstance(simple_system.model, nn.Module)
    assert simple_system.optimizer is not None
    assert simple_system.scheduler is not None
    assert simple_system.criterion is not None
    assert simple_system.metrics.train is not None
    assert simple_system.metrics.val is not None
    assert simple_system.metrics.test is not None


def test_configure_optimizers(simple_system):
    """
    Tests that configure_optimizers returns the correct structure:
    {
        'optimizer': ...,
        'lr_scheduler': ...
    }
    """
    opt_config = simple_system.configure_optimizers()
    assert isinstance(opt_config, dict), "configure_optimizers should return a dictionary."
    assert "optimizer" in opt_config, "Optimizer key missing in configure_optimizers output."
    assert "lr_scheduler" in opt_config, "LR scheduler key missing in configure_optimizers output."


def test_configure_optimizers_without_scheduler(dummy_dataloaders):
    """Test configure_optimizers when no scheduler is provided."""
    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    system = System(
        model=model,
        dataloaders=dummy_dataloaders,
        optimizer=optimizer,
        scheduler=None,
        criterion=nn.CrossEntropyLoss(),
        metrics=None,
        adapters=None,
        inferer=None,
    )

    opt_config = system.configure_optimizers()
    assert isinstance(opt_config, dict)
    assert "optimizer" in opt_config
    assert "lr_scheduler" not in opt_config


def test_on_mode_start_and_end_train(simple_system):
    """Check that _on_mode_start sets the correct mode and _on_mode_end resets it."""
    simple_system._on_mode_start(Mode.TRAIN)
    assert simple_system.mode == Mode.TRAIN
    simple_system._on_mode_end()
    assert simple_system.mode is None


def test_training_step_runs(simple_system):
    """
    Simulate a training step by calling lightning's hooks:
    - on_train_start
    - training_step
    """
    simple_system.on_train_start()
    batch = next(iter(simple_system.dataloaders.train))
    output = simple_system.training_step(batch, batch_idx=0)

    assert isinstance(output, dict), "Expected a dictionary output."
    assert Data.LOSS in output, "Loss should be in output for training mode."
    assert output[Data.LOSS] is not None, "Loss must not be None in train mode."
    assert Data.METRICS in output, "Metrics should be in output for training mode."
    assert Data.PRED in output, "Prediction tensor must be in the output."
    assert output[Data.PRED] is not None, "Pred must not be None."
    assert simple_system.mode == Mode.TRAIN

    simple_system.on_train_end()
    assert simple_system.mode is None


def test_validation_step_runs(simple_system):
    """
    Simulate a validation step by calling:
    - on_validation_start
    - validation_step
    """
    simple_system.on_validation_start()
    batch = next(iter(simple_system.dataloaders.val))
    output = simple_system.validation_step(batch, batch_idx=0)

    assert isinstance(output, dict), "Expected a dictionary output."
    assert Data.LOSS in output, "Loss should be in output for validation mode."
    assert output[Data.LOSS] is not None, "Loss must not be None in validation mode."
    assert Data.METRICS in output, "Metrics should be in output for validation mode."
    assert Data.PRED in output, "Prediction tensor must be in the output."

    simple_system.on_validation_end()
    assert simple_system.mode is None


def test_test_step_runs(simple_system):
    """
    Simulate a test step by calling:
    - on_test_start
    - test_step
    """
    simple_system.on_test_start()
    batch = next(iter(simple_system.dataloaders.test))
    output = simple_system.test_step(batch, batch_idx=0)

    assert isinstance(output, dict), "Expected a dictionary output."
    assert Data.LOSS in output, "Loss should be in output for test mode."
    assert output[Data.LOSS] is None, "Loss must be None in test mode."
    assert Data.METRICS in output, "Metrics should be in output for test mode."
    assert Data.PRED in output, "Prediction tensor must be in the output."

    simple_system.on_test_end()
    assert simple_system.mode is None


def test_predict_step_runs(simple_system):
    """
    Simulate a predict step using the predict_dataloader and check outputs.
    """
    simple_system.on_predict_start()
    batch = next(iter(simple_system.dataloaders.predict))
    output = simple_system.predict_step(batch, batch_idx=0)

    assert isinstance(output, dict), "Expected a dictionary output."
    assert Data.PRED in output, "Predict should contain PRED."
    assert output.get(Data.METRICS) is None, "Metrics must be None in predict mode."
    assert output.get(Data.LOSS) is None, "Loss must be None in predict mode."

    simple_system.on_predict_end()
    assert simple_system.mode is None


def test_no_criterion_in_train_raises_error(simple_system):
    """
    If no criterion is specified in train mode, training_step should raise ValueError.
    """
    # Explicitly set criterion to None
    simple_system.criterion = None

    simple_system.on_train_start()
    batch = next(iter(simple_system.dataloaders.train))
    with pytest.raises(ValueError, match="Please specify 'system.criterion'"):
        _ = simple_system.training_step(batch, 0)


class DictLossNoTotal(nn.Module):
    """
    A module-based "loss" returning a dict with no "total" key to trigger the ValueError.
    """

    def forward(self, pred, target):
        return {"not_total": torch.tensor(1.0)}


def test_dict_loss_without_total_raises_error(simple_system):
    """
    If the criterion returns a dictionary but does not contain 'total' key,
    it should raise ValueError.
    """
    simple_system.criterion = DictLossNoTotal()
    simple_system.on_train_start()

    batch = next(iter(simple_system.dataloaders.train))
    with pytest.raises(ValueError, match="must include a 'total' key"):
        _ = simple_system.training_step(batch, 0)


def test_learning_rate_property(simple_system):
    """Check that learning_rate getter/setter works properly with a single param group."""
    initial_lr = simple_system.learning_rate
    assert initial_lr == 0.01

    simple_system.learning_rate = 0.005
    assert simple_system.learning_rate == 0.005


def test_learning_rate_multiple_param_groups_raises():
    """Ensure accessing .learning_rate with multiple param groups raises ValueError"""
    model = SimpleModel()
    param_groups = [
        {"params": model.linear.weight, "lr": 0.01},
        {"params": model.linear.bias, "lr": 0.001},
    ]
    optimizer = SGD(param_groups)
    system = System(
        model=model,
        dataloaders={"train": DataLoader(DummyDataset())},
        optimizer=optimizer,
        scheduler=None,
        criterion=nn.CrossEntropyLoss(),
        metrics=None,
        adapters=None,
        inferer=None,
    )
    system.trainer = pl.Trainer(logger=False, enable_checkpointing=False, max_epochs=1)
    system.log = MagicMock()

    with pytest.raises(ValueError, match="multiple optimizer parameter groups"):
        _ = system.learning_rate

    with pytest.raises(ValueError, match="multiple optimizer parameter groups"):
        system.learning_rate = 0.0001


def test_inferer_called_in_validation(simple_system):
    """Ensure the inferer function is called in validation mode"""
    mock_inferer = MagicMock(return_value=torch.randn(2, 2))
    simple_system.inferer = mock_inferer

    simple_system.on_validation_start()
    batch = next(iter(simple_system.dataloaders.val))
    _ = simple_system.validation_step(batch, batch_idx=0)

    mock_inferer.assert_called_once()


def test_inferer_called_in_test(simple_system):
    """Ensure the inferer function is called in test mode"""
    mock_inferer = MagicMock(return_value=torch.randn(2, 2))
    simple_system.inferer = mock_inferer

    simple_system.on_test_start()
    batch = next(iter(simple_system.dataloaders.test))
    _ = simple_system.test_step(batch, batch_idx=0)

    mock_inferer.assert_called_once()


def test_loss_logging_single_value(simple_system):
    """Ensure loss logging occurs correctly when it's a single tensor"""
    simple_system.on_train_start()
    batch = next(iter(simple_system.dataloaders.train))
    output = simple_system.training_step(batch, batch_idx=0)

    assert Data.LOSS in output
    simple_system._log_stats.assert_called_once_with(output[Data.LOSS], output[Data.METRICS], 0)


def test_loss_logging_dict_values(simple_system):
    """Ensure loss logging occurs correctly when it's a dict of losses"""

    class MultiLoss(nn.Module):
        def forward(self, pred, target):
            return {"total": torch.tensor(1.0), "aux": torch.tensor(0.5)}

    simple_system.criterion = MultiLoss()

    simple_system.on_train_start()
    batch = next(iter(simple_system.dataloaders.train))
    output = simple_system.training_step(batch, batch_idx=0)

    assert "total" in output[Data.LOSS]
    assert "aux" in output[Data.LOSS]
    simple_system._log_stats.assert_called_once_with(output[Data.LOSS], output[Data.METRICS], 0)


def test_metric_logging(simple_system):
    """Ensure metric logging occurs"""
    simple_system.on_train_start()
    batch = next(iter(simple_system.dataloaders.train))
    output = simple_system.training_step(batch, batch_idx=0)

    assert Data.METRICS in output
    simple_system._log_stats.assert_called_once_with(output[Data.LOSS], output[Data.METRICS], 0)


def test_dynamic_mode_hooks(simple_system):
    """Ensure mode hooks attach dynamically"""
    assert simple_system.training_step is not None
    assert simple_system.train_dataloader is not None
    assert simple_system.on_train_start is not None
    assert simple_system.on_train_end is not None

    assert simple_system.validation_step is not None
    assert simple_system.val_dataloader is not None
    assert simple_system.on_validation_start is not None
    assert simple_system.on_validation_end is not None

    assert simple_system.test_step is not None
    assert simple_system.test_dataloader is not None
    assert simple_system.on_test_start is not None
    assert simple_system.on_test_end is not None

    assert simple_system.predict_step is not None
    assert simple_system.predict_dataloader is not None
    assert simple_system.on_predict_start is not None
    assert simple_system.on_predict_end is not None


def test_log_stats_without_logger(simple_system):
    """Test _log_stats when trainer has no logger."""
    # Override the mock to test actual _log_stats behavior
    simple_system._log_stats = System._log_stats.__get__(simple_system)
    simple_system.trainer.logger = None

    # This should not raise any errors and should return early
    simple_system._log_stats(torch.tensor(1.0), None, 0)


def test_log_stats_with_logger(simple_system):
    """Test _log_stats with a logger."""
    # Override the mock to test actual _log_stats behavior
    simple_system._log_stats = System._log_stats.__get__(simple_system)
    simple_system.trainer.logger = MagicMock()
    simple_system.log = MagicMock()

    # Test single loss value
    simple_system.mode = Mode.TRAIN
    simple_system._log_stats(torch.tensor(1.0), None, 0)
    simple_system.log.assert_called()

    # Test dict loss values
    simple_system.log.reset_mock()
    loss_dict = {"total": torch.tensor(1.0), "aux": torch.tensor(0.5)}
    simple_system._log_stats(loss_dict, None, 0)
    assert simple_system.log.call_count >= 2  # At least one call per loss

    # Test metrics
    simple_system.log.reset_mock()
    metrics = {"accuracy": torch.tensor(0.95)}
    simple_system._log_stats(None, metrics, 0)
    simple_system.log.assert_called()

    # Test optimizer stats (only in train mode, batch_idx=0)
    simple_system.log.reset_mock()
    simple_system._log_stats(None, None, 0)
    simple_system.log.assert_called()  # Should log optimizer stats
