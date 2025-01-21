from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torchmetrics import Accuracy

from lighter.system import LighterSystem


class DummyDataset(Dataset):
    def __init__(self, size=100, return_id=False):
        self.size = size
        self.return_id = return_id

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(3, 32, 32)
        y = torch.randint(0, 10, size=()).long()

        data = {"input": x, "target": y}
        if self.return_id:
            data["id"] = f"id_{idx}"
        return data


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3072, 10))

    def forward(self, x):
        return self.net(x)


class DictCriterion(nn.Module):
    def forward(self, pred, target=None):
        return {"total": torch.nn.functional.cross_entropy(pred, target), "aux": torch.tensor(0.5, requires_grad=True)}


class InvalidDictCriterion(nn.Module):
    def forward(self, pred, target=None):
        return {"aux": torch.tensor(0.5, requires_grad=True)}


@pytest.fixture
def base_system():
    model = DummyModel()
    optimizer = Adam(model.parameters(), lr=0.001)
    system = LighterSystem(
        model=model, batch_size=8, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), datasets={"train": DummyDataset()}
    )
    # Set up minimal trainer
    trainer = Trainer(max_epochs=1)
    system.trainer = trainer
    return system


def test_step_with_dict_loss():
    model = DummyModel()
    criterion = DictCriterion()
    system = LighterSystem(
        model=model, batch_size=8, optimizer=Adam(model.parameters()), criterion=criterion, datasets={"train": DummyDataset()}
    )

    # Mock trainer
    system.trainer = Mock()
    system.trainer.logger = None

    batch = {"input": torch.randn(2, 3, 32, 32), "target": torch.randint(0, 10, (2,))}

    output = system._base_step(batch, 0, "train")
    assert isinstance(output["loss"], torch.Tensor)


def test_step_with_invalid_dict_loss():
    model = DummyModel()
    criterion = InvalidDictCriterion()
    system = LighterSystem(
        model=model, batch_size=8, optimizer=Adam(model.parameters()), criterion=criterion, datasets={"train": DummyDataset()}
    )

    # Mock trainer
    system.trainer = Mock()
    system.trainer.logger = None

    batch = {"input": torch.randn(2, 3, 32, 32), "target": torch.randint(0, 10, (2,))}

    with pytest.raises(ValueError, match="The loss dictionary must include a 'total' key"):
        system._base_step(batch, 0, "train")


def test_invalid_batch_values(base_system):
    # Test None values in optional fields
    invalid_batch = {"input": torch.randn(1, 3, 32, 32), "target": None}
    with pytest.raises(ValueError, match="Batch's 'target' value cannot be None"):
        base_system._base_step(invalid_batch, 0, "train")

    invalid_batch = {"input": torch.randn(1, 3, 32, 32), "id": None}
    with pytest.raises(ValueError, match="Batch's 'id' value cannot be None"):
        base_system._base_step(invalid_batch, 0, "train")


def test_forward_pass(base_system):
    input_tensor = torch.randn(1, 3, 32, 32)
    output = base_system(input_tensor)
    assert output.shape == (1, 10)


def test_learning_rate_property(base_system):
    # Test getter
    assert base_system.learning_rate == 0.001

    # Test setter
    base_system.learning_rate = 0.01
    assert base_system.learning_rate == 0.01
    assert base_system.optimizer.param_groups[0]["lr"] == 0.01


def test_log_stats_without_logger(base_system):
    base_system.trainer.logger = None
    # Should not raise any errors when logger is None
    base_system._log_stats(torch.tensor(1.0), None, "train", 0)


def test_configure_optimizers_variations(base_system):
    # Test with no scheduler
    config = base_system.configure_optimizers()
    assert "optimizer" in config
    assert "lr_scheduler" not in config

    # Test with scheduler
    base_system.scheduler = StepLR(base_system.optimizer, step_size=1)
    config = base_system.configure_optimizers()
    assert "optimizer" in config
    assert "lr_scheduler" in config


def test_setup_stages(base_system):
    # Test fit stage
    base_system.setup("fit")
    assert callable(base_system.train_dataloader)
    assert callable(base_system.training_step)

    # Test validate stage
    base_system.setup("validate")
    assert callable(base_system.val_dataloader)
    assert callable(base_system.validation_step)

    # Test test stage
    base_system.setup("test")
    assert callable(base_system.test_dataloader)
    assert callable(base_system.test_step)

    # Test predict stage
    base_system.setup("predict")
    assert callable(base_system.predict_dataloader)
    assert callable(base_system.predict_step)


def test_forward_with_step_and_epoch():
    class ModelWithStepEpoch(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x, step=None, epoch=None):
            # Add assertions to verify step/epoch are passed correctly
            assert step is not None, "step should not be None"
            assert epoch is not None, "epoch should not be None"
            return x

    # Set up model and system
    model = ModelWithStepEpoch()
    system = LighterSystem(
        model=model,
        batch_size=8,
        optimizer=Adam(model.parameters()),
        criterion=nn.MSELoss(),
        datasets={"train": DummyDataset()},
    )

    # Set up mock trainer with required attributes
    mock_trainer = Mock()
    mock_trainer.global_step = 1
    mock_trainer.current_epoch = 2
    system.trainer = mock_trainer

    # Test forward pass with input
    x = torch.randn(1, 3, 32, 32)
    _ = system(x)  # Should work without error


def test_setup_without_datasets_with_error():
    model = DummyModel()

    # Create system without any datasets
    system = LighterSystem(
        model=model,
        batch_size=8,
        optimizer=Adam(model.parameters()),
        datasets={},  # Explicitly pass empty dict to ensure no datasets
    )

    # Setup will succeed, but trying to access the dataloader should fail
    # This will trigger the error on line 149
    system.setup("fit")
    with pytest.raises(ValueError, match="Please specify 'train' dataset"):
        # This will trigger _base_dataloader which raises the error
        _ = system.train_dataloader()


def test_batch_without_target_with_error():
    model = DummyModel()
    system = LighterSystem(
        model=model,
        batch_size=8,
        optimizer=Adam(model.parameters()),
        criterion=lambda x: torch.sum(x),  # Criterion that only uses predictions
        datasets={"train": DummyDataset()},
    )

    system.trainer = Mock()
    batch = {"input": torch.randn(2, 3, 32, 32)}
    output = system._base_step(batch, 0, "train")
    assert "loss" in output


def test_batch_type_validation():
    model = DummyModel()
    system = LighterSystem(model=model, batch_size=8, optimizer=Adam(model.parameters()), datasets={"train": DummyDataset()})

    system.trainer = Mock()

    # Test non-dict batch
    with pytest.raises(TypeError, match="Batch must be a dict"):
        system._base_step([torch.randn(2, 3, 32, 32)], 0, "train")

    # Test invalid keys
    with pytest.raises(ValueError, match="Batch must be a dict with keys"):
        system._base_step({"wrong_key": torch.randn(2, 3, 32, 32)}, 0, "train")


def test_batch_processing_stages():
    def count_calls(x):
        count_calls.calls += 1
        return x

    count_calls.calls = 0

    model = DummyModel()
    system = LighterSystem(
        model=model,
        batch_size=8,
        optimizer=Adam(model.parameters()),
        criterion=nn.CrossEntropyLoss(),
        datasets={"train": DummyDataset()},
        postprocessing={
            "batch": {"train": count_calls},
            "criterion": {"input": None, "target": None, "pred": None},
            "metrics": {"input": None, "target": None, "pred": None},
            "logging": {"input": None, "target": None, "pred": None},
        },
    )

    # Mock trainer
    mock_trainer = Mock()
    mock_trainer.global_step = 0
    system.trainer = mock_trainer

    # Test batch processing
    batch = {"input": torch.randn(2, 3, 32, 32), "target": torch.randint(0, 10, (2,))}
    _ = system._base_step(batch, 0, "train")
    assert count_calls.calls == 1


def test_multiple_param_groups():
    model = DummyModel()
    optimizer = Adam([{"params": model.net[0].parameters(), "lr": 0.001}, {"params": model.net[1].parameters(), "lr": 0.002}])

    system = LighterSystem(model=model, batch_size=8, optimizer=optimizer, datasets={"train": DummyDataset()})

    with pytest.raises(ValueError, match="multiple optimizer parameter groups"):
        _ = system.learning_rate

    with pytest.raises(ValueError, match="multiple optimizer parameter groups"):
        system.learning_rate = 0.1


def test_configure_optimizers_no_optimizer():
    model = DummyModel()
    system = LighterSystem(model=model, batch_size=8, criterion=nn.CrossEntropyLoss(), datasets={"train": DummyDataset()})

    with pytest.raises(ValueError, match="Please specify 'system.optimizer' in the config."):
        system.configure_optimizers()

    model = DummyModel()
    system = LighterSystem(
        model=model,
        batch_size=8,
        optimizer=Adam(model.parameters()),
        datasets={"train": DummyDataset()},
        metrics={"train": Accuracy(task="multiclass", num_classes=10)},
    )

    # Mock trainer with logger
    system.trainer = Mock()
    system.trainer.logger = Mock()

    # Test logging loss
    loss = torch.tensor(1.0)
    metrics = {"accuracy": torch.tensor(0.8)}
    system._log_stats(loss, metrics, "train", 0)

    # Test dict loss logging
    dict_loss = {"total": torch.tensor(1.0), "component": torch.tensor(0.5)}
    system._log_stats(dict_loss, metrics, "train", 0)
