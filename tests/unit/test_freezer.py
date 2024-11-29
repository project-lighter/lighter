import pytest
from lighter.callbacks.freezer import LighterFreezer
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from torch.nn import Module

class DummyModel(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)

def test_freezer_initialization():
    freezer = LighterFreezer(names=["layer1"])
    assert freezer.names == ["layer1"]

class DummyDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return torch.randn(10), torch.tensor(0)

def test_freezer_with_trainer():
    model = DummyModel()
    freezer = LighterFreezer(names=["layer1"])
    trainer = Trainer(callbacks=[freezer], max_epochs=1)
    trainer.fit_loop.setup_data(DataLoader(DummyDataset()))
    freezer.on_train_batch_start(trainer, None, None, 0)
    assert not model.layer1.weight.requires_grad
    assert model.layer2.weight.requires_grad
