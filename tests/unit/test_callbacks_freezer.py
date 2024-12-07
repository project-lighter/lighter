import pytest
import torch
from pytorch_lightning import Trainer
from torch.nn import Module
from torch.utils.data import Dataset

from lighter.callbacks.freezer import LighterFreezer
from lighter.system import LighterSystem


class DummyModel(Module):
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


class DummyDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return {"input": torch.randn(10), "target": torch.tensor(0)}


@pytest.fixture
def dummy_system():
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dataset = DummyDataset()
    criterion = torch.nn.CrossEntropyLoss()
    return LighterSystem(model=model, batch_size=8, criterion=criterion, optimizer=optimizer, datasets={"train": dataset})


def test_freezer_initialization():
    freezer = LighterFreezer(names=["layer1"])
    assert freezer.names == ["layer1"]


def test_freezer_functionality(dummy_system):
    freezer = LighterFreezer(names=["layer1.weight", "layer1.bias"])
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert not dummy_system.model.layer1.weight.requires_grad
    assert not dummy_system.model.layer1.bias.requires_grad
    assert dummy_system.model.layer2.weight.requires_grad


def test_freezer_with_exceptions(dummy_system):
    freezer = LighterFreezer(name_starts_with=["layer"], except_names=["layer2.weight", "layer2.bias"])
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert not dummy_system.model.layer1.weight.requires_grad
    assert not dummy_system.model.layer1.bias.requires_grad
    assert dummy_system.model.layer2.weight.requires_grad
    assert dummy_system.model.layer2.bias.requires_grad
    assert not dummy_system.model.layer3.weight.requires_grad
    assert not dummy_system.model.layer3.bias.requires_grad
