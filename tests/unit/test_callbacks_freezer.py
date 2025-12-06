import pytest
import torch
from pytorch_lightning import Trainer
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from lighter.callbacks.freezer import Freezer
from lighter.model import LighterModule


class DummyDataset(Dataset):
    """Simple dataset for testing."""

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
    """Three-layer network for testing freezing behavior."""

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


class DummyLighterModule(LighterModule):
    """Concrete System implementation for testing."""

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return {"pred": pred, "target": y}

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def train_dataloader(self):
        return DataLoader(DummyDataset(), batch_size=32)

    def val_dataloader(self):
        return DataLoader(DummyDataset(), batch_size=32)


@pytest.fixture
def dummy_system():
    """Create a LighterModule with DummyModel for freezer tests."""
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    return DummyLighterModule(
        network=model,
        criterion=criterion,
        optimizer=optimizer,
    )


def test_freezer_initialization():
    """Test Freezer initialization validates parameters correctly."""
    with pytest.raises(ValueError, match="At least one of `names` or `name_starts_with` must be specified."):
        Freezer()

    with pytest.raises(ValueError, match="Only one of `until_step` or `until_epoch` can be specified."):
        Freezer(names=["layer1"], until_step=10, until_epoch=1)
    freezer = Freezer(names=["layer1"])
    assert freezer.names == ["layer1"]


def test_freezer_functionality(dummy_system):
    """Test that specified layers are frozen while others remain trainable."""
    freezer = Freezer(names=["layer1.weight", "layer1.bias"])
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert not dummy_system.network.layer1.weight.requires_grad
    assert not dummy_system.network.layer1.bias.requires_grad
    assert dummy_system.network.layer2.weight.requires_grad


def test_freezer_exceed_until_step(dummy_system):
    """Test that layers are unfrozen after exceeding until_step."""
    freezer = Freezer(names=["layer1.weight", "layer1.bias"], until_step=0)
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert dummy_system.network.layer1.weight.requires_grad
    assert dummy_system.network.layer1.bias.requires_grad

    # Test unfreezing after exceeding until_step
    freezer = Freezer(names=["layer1.weight", "layer1.bias"], until_step=1)
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert dummy_system.network.layer1.weight.requires_grad
    assert dummy_system.network.layer1.bias.requires_grad


def test_freezer_exceed_until_epoch(dummy_system):
    """Test that layers are unfrozen after exceeding until_epoch."""
    freezer = Freezer(names=["layer1.weight", "layer1.bias"], until_epoch=0)
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert dummy_system.network.layer1.weight.requires_grad
    assert dummy_system.network.layer1.bias.requires_grad

    # Test unfreezing after exceeding until_epoch
    freezer = Freezer(names=["layer1.weight", "layer1.bias"], until_epoch=1)
    trainer = Trainer(callbacks=[freezer], max_epochs=2)
    trainer.fit(dummy_system)
    assert dummy_system.network.layer1.weight.requires_grad
    assert dummy_system.network.layer1.bias.requires_grad


def test_freezer_set_model_requires_grad(dummy_system):
    """Test _set_model_requires_grad freezes and unfreezes parameters."""
    freezer = Freezer(names=["layer1.weight", "layer1.bias"])
    freezer._set_model_requires_grad(dummy_system.network, requires_grad=False)
    assert not dummy_system.network.layer1.weight.requires_grad
    assert not dummy_system.network.layer1.bias.requires_grad
    freezer._set_model_requires_grad(dummy_system.network, requires_grad=True)
    assert dummy_system.network.layer1.weight.requires_grad
    assert dummy_system.network.layer1.bias.requires_grad

    # Test with exceptions
    freezer = Freezer(names=["layer1.weight", "layer1.bias"], except_names=["layer1.bias"])
    freezer._set_model_requires_grad(dummy_system.network, requires_grad=False)
    assert not dummy_system.network.layer1.weight.requires_grad
    assert dummy_system.network.layer1.bias.requires_grad
    freezer._set_model_requires_grad(dummy_system.network, requires_grad=True)
    assert dummy_system.network.layer1.weight.requires_grad
    assert dummy_system.network.layer1.bias.requires_grad


def test_freezer_with_exceptions(dummy_system):
    """Test Freezer respects except_names and except_name_starts_with."""
    freezer = Freezer(name_starts_with=["layer"], except_names=["layer2.weight", "layer2.bias"])
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert not dummy_system.network.layer1.weight.requires_grad
    assert not dummy_system.network.layer1.bias.requires_grad
    assert dummy_system.network.layer2.weight.requires_grad
    assert dummy_system.network.layer2.bias.requires_grad
    assert not dummy_system.network.layer3.weight.requires_grad
    assert not dummy_system.network.layer3.bias.requires_grad

    # Test with except_name_starts_with
    freezer = Freezer(name_starts_with=["layer"], except_name_starts_with=["layer2"])
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert not dummy_system.network.layer1.weight.requires_grad
    assert not dummy_system.network.layer1.bias.requires_grad
    assert dummy_system.network.layer2.weight.requires_grad
    assert dummy_system.network.layer2.bias.requires_grad
    assert not dummy_system.network.layer3.weight.requires_grad
    assert not dummy_system.network.layer3.bias.requires_grad


def test_freezer_except_name_starts_with(dummy_system):
    """Test Freezer with except_name_starts_with parameter."""
    freezer = Freezer(name_starts_with=["layer"], except_name_starts_with=["layer2"])
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert not dummy_system.network.layer1.weight.requires_grad
    assert not dummy_system.network.layer1.bias.requires_grad
    assert dummy_system.network.layer2.weight.requires_grad
    assert dummy_system.network.layer2.bias.requires_grad
    assert not dummy_system.network.layer3.weight.requires_grad
    assert not dummy_system.network.layer3.bias.requires_grad

    # Test with both except_names and except_name_starts_with
    freezer = Freezer(
        name_starts_with=["layer"],
        except_names=["layer2.bias"],
        except_name_starts_with=["layer3"],
    )
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert not dummy_system.network.layer1.weight.requires_grad
    assert not dummy_system.network.layer1.bias.requires_grad
    assert not dummy_system.network.layer2.weight.requires_grad
    assert dummy_system.network.layer2.bias.requires_grad
    assert dummy_system.network.layer3.weight.requires_grad
    assert dummy_system.network.layer3.bias.requires_grad


def test_freezer_set_model_requires_grad_with_exceptions(dummy_system):
    """Test _set_model_requires_grad with various exception patterns."""
    freezer = Freezer(names=["layer1.weight", "layer1.bias"], except_names=["layer1.bias"])
    freezer._set_model_requires_grad(dummy_system.network, requires_grad=False)
    assert not dummy_system.network.layer1.weight.requires_grad
    assert dummy_system.network.layer1.bias.requires_grad
    freezer._set_model_requires_grad(dummy_system.network, requires_grad=True)
    assert dummy_system.network.layer1.weight.requires_grad
    assert dummy_system.network.layer1.bias.requires_grad
    freezer = Freezer(name_starts_with=["layer"], except_names=["layer2.weight", "layer2.bias"])
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert not dummy_system.network.layer1.weight.requires_grad
    assert not dummy_system.network.layer1.bias.requires_grad
    assert dummy_system.network.layer2.weight.requires_grad
    assert dummy_system.network.layer2.bias.requires_grad
    assert not dummy_system.network.layer3.weight.requires_grad
    assert not dummy_system.network.layer3.bias.requires_grad

    # Test with until_step and until_epoch
    freezer = Freezer(names=["layer1.weight", "layer1.bias"], until_step=1)
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert dummy_system.network.layer1.weight.requires_grad
    assert dummy_system.network.layer1.bias.requires_grad

    freezer = Freezer(names=["layer1.weight", "layer1.bias"], until_epoch=1)
    trainer = Trainer(callbacks=[freezer], max_epochs=2)
    trainer.fit(dummy_system)
    assert dummy_system.network.layer1.weight.requires_grad
    assert dummy_system.network.layer1.bias.requires_grad
