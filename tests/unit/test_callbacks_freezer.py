import pytest
import torch
from pytorch_lightning import Trainer
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from lighter.callbacks.freezer import Freezer
from lighter.system import System


class DummyDataset(Dataset):
    """
    A simple dataset for testing purposes.

    Generates random input data and labels for training.
    """

    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, 10)
        # add ", 1" after num_samples and make it .float() to ensure compatibility with BCEWithLogitsLoss
        self.labels = torch.randint(0, 2, (num_samples, 1)).float()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class DummyModel(Module):
    """
    A simple neural network model for testing purposes.

    Contains three linear layers that can be selectively frozen during training.
    """

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 4)
        self.layer3 = torch.nn.Linear(4, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


@pytest.fixture
def dummy_system():
    """
    Fixture that creates a System instance with a dummy model for testing.

    Returns:
        System: A configured system with DummyModel, SGD optimizer, and DummyDataset.
    """
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_dataloader = DataLoader(DummyDataset(), batch_size=32)
    return System(model=model, criterion=criterion, optimizer=optimizer, dataloaders={"train": train_dataloader})


def test_freezer_initialization():
    """
    Test the initialization of Freezer with various parameter combinations.

    Verifies:
        - Raises ValueError when neither names nor name_starts_with is specified
        - Raises ValueError when both until_step and until_epoch are specified
        - Correctly stores the names parameter
    """
    with pytest.raises(ValueError, match="At least one of `names` or `name_starts_with` must be specified."):
        Freezer()

    with pytest.raises(ValueError, match="Only one of `until_step` or `until_epoch` can be specified."):
        Freezer(names=["layer1"], until_step=10, until_epoch=1)
    freezer = Freezer(names=["layer1"])
    assert freezer.names == ["layer1"]


def test_freezer_functionality(dummy_system):
    """
    Test the basic functionality of Freezer during training.

    Verifies:
        - Specified layers are correctly frozen (requires_grad=False)
        - Non-specified layers remain unfrozen (requires_grad=True)
    """
    freezer = Freezer(names=["layer1.weight", "layer1.bias"])
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert not dummy_system.model.layer1.weight.requires_grad
    assert not dummy_system.model.layer1.bias.requires_grad
    assert dummy_system.model.layer2.weight.requires_grad


def test_freezer_exceed_until_step(dummy_system):
    """
    Test that layers are unfrozen after exceeding the specified step limit.

    Verifies that layers become trainable (requires_grad=True) after the until_step threshold.
    """
    freezer = Freezer(names=["layer1.weight", "layer1.bias"], until_step=0)
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert dummy_system.model.layer1.weight.requires_grad
    assert dummy_system.model.layer1.bias.requires_grad

    # Test unfreezing after exceeding until_step
    freezer = Freezer(names=["layer1.weight", "layer1.bias"], until_step=1)
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert dummy_system.model.layer1.weight.requires_grad
    assert dummy_system.model.layer1.bias.requires_grad


def test_freezer_exceed_until_epoch(dummy_system):
    """
    Test that layers are unfrozen after exceeding the specified epoch limit.

    Verifies that layers become trainable (requires_grad=True) after the until_epoch threshold.
    """
    freezer = Freezer(names=["layer1.weight", "layer1.bias"], until_epoch=0)
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert dummy_system.model.layer1.weight.requires_grad
    assert dummy_system.model.layer1.bias.requires_grad

    # Test unfreezing after exceeding until_epoch
    freezer = Freezer(names=["layer1.weight", "layer1.bias"], until_epoch=1)
    trainer = Trainer(callbacks=[freezer], max_epochs=2)
    trainer.fit(dummy_system)
    assert dummy_system.model.layer1.weight.requires_grad
    assert dummy_system.model.layer1.bias.requires_grad


def test_freezer_set_model_requires_grad(dummy_system):
    """
    Test the internal _set_model_requires_grad method of Freezer.

    Verifies:
        - Method correctly freezes specified parameters
        - Method correctly unfreezes specified parameters
    """
    freezer = Freezer(names=["layer1.weight", "layer1.bias"])
    freezer._set_model_requires_grad(dummy_system.model, requires_grad=False)
    assert not dummy_system.model.layer1.weight.requires_grad
    assert not dummy_system.model.layer1.bias.requires_grad
    freezer._set_model_requires_grad(dummy_system.model, requires_grad=True)
    assert dummy_system.model.layer1.weight.requires_grad
    assert dummy_system.model.layer1.bias.requires_grad

    # Test with exceptions
    freezer = Freezer(names=["layer1.weight", "layer1.bias"], except_names=["layer1.bias"])
    freezer._set_model_requires_grad(dummy_system.model, requires_grad=False)
    assert not dummy_system.model.layer1.weight.requires_grad
    assert dummy_system.model.layer1.bias.requires_grad
    freezer._set_model_requires_grad(dummy_system.model, requires_grad=True)
    assert dummy_system.model.layer1.weight.requires_grad
    assert dummy_system.model.layer1.bias.requires_grad


def test_freezer_with_exceptions(dummy_system):
    """
    Test Freezer with exception patterns for layer freezing.

    Verifies:
        - Layers matching name_starts_with are frozen
        - Layers in except_names remain unfrozen
        - Other layers behave as expected
    """
    freezer = Freezer(name_starts_with=["layer"], except_names=["layer2.weight", "layer2.bias"])
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert not dummy_system.model.layer1.weight.requires_grad
    assert not dummy_system.model.layer1.bias.requires_grad
    assert dummy_system.model.layer2.weight.requires_grad
    assert dummy_system.model.layer2.bias.requires_grad
    assert not dummy_system.model.layer3.weight.requires_grad
    assert not dummy_system.model.layer3.bias.requires_grad

    # Test with except_name_starts_with
    freezer = Freezer(name_starts_with=["layer"], except_name_starts_with=["layer2"])
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert not dummy_system.model.layer1.weight.requires_grad
    assert not dummy_system.model.layer1.bias.requires_grad
    assert dummy_system.model.layer2.weight.requires_grad
    assert dummy_system.model.layer2.bias.requires_grad
    assert not dummy_system.model.layer3.weight.requires_grad
    assert not dummy_system.model.layer3.bias.requires_grad


def test_freezer_except_name_starts_with(dummy_system):
    """
    Test Freezer with except_name_starts_with parameter.

    Verifies:
        - Layers matching name_starts_with are frozen
        - Layers matching except_name_starts_with remain unfrozen
        - Other layers behave as expected
    """
    freezer = Freezer(name_starts_with=["layer"], except_name_starts_with=["layer2"])
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert not dummy_system.model.layer1.weight.requires_grad
    assert not dummy_system.model.layer1.bias.requires_grad
    assert dummy_system.model.layer2.weight.requires_grad
    assert dummy_system.model.layer2.bias.requires_grad
    assert not dummy_system.model.layer3.weight.requires_grad
    assert not dummy_system.model.layer3.bias.requires_grad

    # Test with both except_names and except_name_starts_with
    freezer = Freezer(
        name_starts_with=["layer"],
        except_names=["layer2.bias"],
        except_name_starts_with=["layer3"],
    )
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert not dummy_system.model.layer1.weight.requires_grad
    assert not dummy_system.model.layer1.bias.requires_grad
    assert not dummy_system.model.layer2.weight.requires_grad
    assert dummy_system.model.layer2.bias.requires_grad
    assert dummy_system.model.layer3.weight.requires_grad
    assert dummy_system.model.layer3.bias.requires_grad


def test_freezer_set_model_requires_grad_with_exceptions(dummy_system):
    """
    Test the _set_model_requires_grad method with various exception patterns.

    Verifies:
        - Correct handling of specific parameter exceptions
        - Proper behavior with name_starts_with and except_names combinations
        - Consistent freezing/unfreezing across multiple configurations
    """
    freezer = Freezer(names=["layer1.weight", "layer1.bias"], except_names=["layer1.bias"])
    freezer._set_model_requires_grad(dummy_system.model, requires_grad=False)
    assert not dummy_system.model.layer1.weight.requires_grad
    assert dummy_system.model.layer1.bias.requires_grad
    freezer._set_model_requires_grad(dummy_system.model, requires_grad=True)
    assert dummy_system.model.layer1.weight.requires_grad
    assert dummy_system.model.layer1.bias.requires_grad
    freezer = Freezer(name_starts_with=["layer"], except_names=["layer2.weight", "layer2.bias"])
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert not dummy_system.model.layer1.weight.requires_grad
    assert not dummy_system.model.layer1.bias.requires_grad
    assert dummy_system.model.layer2.weight.requires_grad
    assert dummy_system.model.layer2.bias.requires_grad
    assert not dummy_system.model.layer3.weight.requires_grad
    assert not dummy_system.model.layer3.bias.requires_grad

    # Test with until_step and until_epoch
    freezer = Freezer(names=["layer1.weight", "layer1.bias"], until_step=1)
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert dummy_system.model.layer1.weight.requires_grad
    assert dummy_system.model.layer1.bias.requires_grad

    freezer = Freezer(names=["layer1.weight", "layer1.bias"], until_epoch=1)
    trainer = Trainer(callbacks=[freezer], max_epochs=2)
    trainer.fit(dummy_system)
    assert dummy_system.model.layer1.weight.requires_grad
    assert dummy_system.model.layer1.bias.requires_grad
