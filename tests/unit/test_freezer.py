import pytest
from lighter.callbacks.freezer import LighterFreezer
import torch
from torch.nn import Module
from lighter.system import LighterSystem
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer

class DummyModel(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)

class DummyDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return torch.randn(10), torch.tensor(0)

class DummySystem(LighterSystem):
    def __init__(self):
        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        super().__init__(model=model, batch_size=32, optimizer=optimizer, datasets={"train": DummyDataset()})

@pytest.fixture
def dummy_system():
    return DummySystem()

def test_freezer_initialization():
    freezer = LighterFreezer(names=["layer1"])
    assert freezer.names == ["layer1"]

def test_freezer_functionality(dummy_system):
    freezer = LighterFreezer(names=["layer1"])
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert not dummy_system.model.layer1.weight.requires_grad
    assert dummy_system.model.layer2.weight.requires_grad

def test_freezer_with_exceptions(dummy_system):
    freezer = LighterFreezer(names=["layer1"], except_names=["layer1.weight"])
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit(dummy_system)
    assert dummy_system.model.layer1.weight.requires_grad
    assert not dummy_system.model.layer1.bias.requires_grad
    assert dummy_system.model.layer2.weight.requires_grad
