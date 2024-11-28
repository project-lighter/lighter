import pytest
from lighter.callbacks.freezer import LighterFreezer
from torch.nn import Module

class DummyModel(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)

def test_freezer_initialization():
    freezer = LighterFreezer(names=["layer1"])
    assert freezer.names == ["layer1"]

def test_freezer_freezing():
    model = DummyModel()
    freezer = LighterFreezer(names=["layer1"])
    freezer._set_model_requires_grad(model, False)
    assert not model.layer1.weight.requires_grad
    assert model.layer2.weight.requires_grad
