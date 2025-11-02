"""Unit tests for the System class."""

from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy

from lighter.system import System
from lighter.utils.types.enums import Mode


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


@pytest.fixture
def model():
    """Returns a simple model."""
    return nn.Linear(4, 2)


@pytest.fixture
def optimizer(model):
    """Returns a simple optimizer."""
    return SGD(model.parameters(), lr=0.01)


@pytest.fixture
def scheduler(optimizer):
    """Returns a simple scheduler."""
    return StepLR(optimizer, step_size=10, gamma=0.1)


@pytest.fixture
def criterion():
    """Returns a simple criterion."""
    return nn.CrossEntropyLoss()


@pytest.fixture
def metrics():
    """Returns a simple metrics collection."""
    return {
        "train": Accuracy(task="multiclass", num_classes=2),
        "val": Accuracy(task="multiclass", num_classes=2),
        "test": Accuracy(task="multiclass", num_classes=2),
    }


@pytest.fixture
def dataloaders():
    """Provides train/val/test/predict DataLoaders"""
    return {
        "train": DataLoader(DummyDataset(size=8, with_target=True), batch_size=2),
        "val": DataLoader(DummyDataset(size=4, with_target=True), batch_size=2),
        "test": DataLoader(DummyDataset(size=4, with_target=True), batch_size=2),
        "predict": DataLoader(DummyDataset(size=4, with_target=False), batch_size=2),
    }


@pytest.fixture
def system(model, optimizer, scheduler, criterion, metrics, dataloaders):
    """Creates a System instance."""
    system = System(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        metrics=metrics,
        dataloaders=dataloaders,
    )
    system.trainer = pl.Trainer(logger=None, enable_checkpointing=False, max_epochs=1)
    return system


def test_system_initialization(system):
    """Check that attributes are correctly set after initialization."""
    assert isinstance(system.model, nn.Module)
    assert system.optimizer is not None
    assert system.scheduler is not None
    assert system.criterion is not None
    assert system.metrics["train"] is not None
    assert system.metrics["val"] is not None
    assert system.metrics["test"] is not None


def test_configure_optimizers(system):
    """
    Tests that configure_optimizers returns the correct structure.
    """
    opt_config = system.configure_optimizers()
    assert isinstance(opt_config, dict), "configure_optimizers should return a dictionary."
    assert "optimizer" in opt_config, "Optimizer key missing in configure_optimizers output."
    assert "lr_scheduler" in opt_config, "LR scheduler key missing in configure_optimizers output."


def test_step(system):
    """Test the `_step` method directly."""
    system.mode = Mode.TRAIN
    batch = next(iter(system.dataloaders.train))
    output = system._step(batch, 0)
    assert "loss" in output
    assert "pred" in output


def test_log_stats(system):
    """Test the `_log_stats` method."""
    system.log = MagicMock()
    system.mode = Mode.TRAIN
    # Test with loss
    system._log_stats({"loss": torch.tensor(0.1)}, 0)
    assert system.log.call_count == 4  # loss step, loss epoch, lr, momentum
    system.log.reset_mock()
    # Test with metrics
    system._log_stats({"metrics": {"acc": torch.tensor(0.9)}}, 0)
    assert system.log.call_count == 4  # metric step, metric epoch, lr, momentum


def test_learning_rate_property(system):
    """Test the learning_rate property."""
    # Test getter
    assert system.learning_rate == 0.01

    # Test setter
    system.learning_rate = 0.05
    assert system.learning_rate == 0.05
    assert system.optimizer.param_groups[0]["lr"] == 0.05


def test_learning_rate_property_multiple_param_groups(model, criterion, dataloaders):
    """Test that the learning_rate property raises ValueError with multiple param groups."""
    optimizer = SGD(
        [{"params": model.parameters(), "lr": 0.01}, {"params": [torch.nn.Parameter(torch.randn(2, 2))], "lr": 0.1}]
    )
    system = System(model=model, optimizer=optimizer, criterion=criterion, dataloaders=dataloaders)

    with pytest.raises(ValueError):
        _ = system.learning_rate

    with pytest.raises(ValueError):
        system.learning_rate = 0.05
