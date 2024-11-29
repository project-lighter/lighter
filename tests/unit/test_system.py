from unittest import mock

import pytest
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy

from lighter.system import LighterSystem


class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(3, 32, 32)

        y = torch.randint(0, 10, size=()).long()  # Changed to return scalar tensor
        return {"input": x, "target": y}


class DummyPredictDataset(Dataset):
    def __init__(self, size=20):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(3, 32, 32)
        return {"input": x}


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3072, 10))

    def forward(self, x):
        return self.net(x)


class DummySystem(LighterSystem):
    def __init__(self):
        model = DummyModel()
        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=1)
        criterion = nn.CrossEntropyLoss()

        datasets = {
            "train": DummyDataset(),
            "val": DummyDataset(50),
            "test": DummyDataset(20),
            "predict": DummyPredictDataset(20),
        }

        metrics = {
            "train": Accuracy(task="multiclass", num_classes=10),
            "val": Accuracy(task="multiclass", num_classes=10),
            "test": Accuracy(task="multiclass", num_classes=10),
        }

        super().__init__(
            model=model,
            batch_size=32,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            datasets=datasets,
            metrics=metrics,
        )


@pytest.fixture
def dummy_system():
    return DummySystem()


def test_system_with_trainer(dummy_system):
    trainer = Trainer(max_epochs=1)
    trainer.fit(dummy_system)
    assert dummy_system.batch_size == 32
    assert isinstance(dummy_system.model, DummyModel)
    assert isinstance(dummy_system.optimizer, Adam)
    assert isinstance(dummy_system.scheduler, StepLR)


def test_configure_optimizers(dummy_system):
    config = dummy_system.configure_optimizers()
    assert "optimizer" in config
    assert "lr_scheduler" in config
    assert isinstance(config["optimizer"], Adam)
    assert isinstance(config["lr_scheduler"], StepLR)


def test_dataloader_creation(dummy_system):
    dummy_system.setup("fit")
    train_loader = dummy_system.train_dataloader()
    assert isinstance(train_loader, DataLoader)
    assert train_loader.batch_size == 32


def test_training_step(dummy_system):
    trainer = Trainer(max_epochs=1)
    trainer.fit(dummy_system)  # Only to attach the system to the trainer
    batch = next(iter(dummy_system.train_dataloader()))

    # https://github.com/Lightning-AI/pytorch-lightning/issues/9674#issuecomment-926243063
    with mock.patch.object(dummy_system, "log"):
        result = dummy_system.training_step(batch, batch_idx=0)

    assert "loss" in result
    assert "metrics" in result
    assert "input" in result
    assert "target" in result
    assert "pred" in result
    assert torch.is_tensor(result["loss"])


def test_validation_step(dummy_system):
    trainer = Trainer(max_epochs=1)
    trainer.validate(dummy_system)  # Only to attach the system to the trainer
    batch = next(iter(dummy_system.val_dataloader()))

    # https://github.com/Lightning-AI/pytorch-lightning/issues/9674#issuecomment-926243063
    with mock.patch.object(dummy_system, "log"):
        result = dummy_system.validation_step(batch, batch_idx=0)

    assert "loss" in result
    assert "metrics" in result
    assert torch.is_tensor(result["loss"])


def test_predict_step(dummy_system):
    trainer = Trainer(max_epochs=1)
    trainer.predict(dummy_system)  # Only to attach the system to the trainer
    batch = next(iter(dummy_system.predict_dataloader()))

    # https://github.com/Lightning-AI/pytorch-lightning/issues/9674#issuecomment-926243063
    with mock.patch.object(dummy_system, "log"):
        result = dummy_system.predict_step(batch, batch_idx=0)

    assert "pred" in result
    assert torch.is_tensor(result["pred"])


def test_learning_rate_property(dummy_system):
    initial_lr = dummy_system.learning_rate
    assert initial_lr == 0.001

    dummy_system.learning_rate = 0.01
    assert dummy_system.learning_rate == 0.01


@pytest.mark.parametrize(
    "batch",
    [
        {"input": torch.randn(1, 3, 32, 32), "target": torch.randint(0, 10, size=(1,)).long()},
        {"input": torch.randn(2, 3, 32, 32), "target": torch.randint(0, 10, size=(2,)).long()},
        {"input": torch.randn(4, 3, 32, 32), "target": torch.randint(0, 10, size=(4,)).long(), "id": "test_id"},
    ],
)
def test_valid_batch_formats(dummy_system, batch):
    trainer = Trainer(max_epochs=1)
    trainer.fit(dummy_system)  # Only to attach the system to the trainer

    # https://github.com/Lightning-AI/pytorch-lightning/issues/9674#issuecomment-926243063
    with mock.patch.object(dummy_system, "log"):
        result = dummy_system.training_step(batch, batch_idx=0)
    assert isinstance(result, dict)


@pytest.mark.xfail(raises=ValueError)
def test_invalid_batch_format(dummy_system):
    invalid_batch = {"wrong_key": torch.randn(1, 3, 32, 32)}
    trainer = Trainer(max_epochs=1)
    trainer.fit(dummy_system)  # Only to attach the system to the trainer

    # https://github.com/Lightning-AI/pytorch-lightning/issues/9674#issuecomment-926243063
    with mock.patch.object(dummy_system, "log"):
        dummy_system.training_step(invalid_batch, batch_idx=0)
