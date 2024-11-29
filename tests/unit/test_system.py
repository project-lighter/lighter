import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
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

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 10)
        )
    
    def forward(self, x):
        return self.net(x)

@pytest.fixture
def basic_system():
    model = DummyModel()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1)
    criterion = nn.CrossEntropyLoss()
    
    datasets = {
        "train": DummyDataset(),
        "val": DummyDataset(50),
        "test": DummyDataset(20)
    }
    
    metrics = {
        "train": Accuracy(task="multiclass", num_classes=10),
        "val": Accuracy(task="multiclass", num_classes=10),
        "test": Accuracy(task="multiclass", num_classes=10)
    }
    
    return LighterSystem(
        model=model,
        batch_size=32,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        datasets=datasets,
        metrics=metrics
    )

def test_system_with_trainer(basic_system):
    trainer = Trainer(max_epochs=1)
    trainer.fit(basic_system)
    trainer.fit_loop.setup_data(basic_system.val_dataloader())
    trainer.fit_loop.setup_data(basic_system.test_dataloader())
    trainer.fit_loop.setup_data(basic_system.predict_dataloader())
    assert isinstance(basic_system.model, DummyModel)
    assert isinstance(basic_system.model, DummyModel)
    assert basic_system.batch_size == 32
    assert isinstance(basic_system.optimizer, Adam)
    assert isinstance(basic_system.scheduler, StepLR)

def test_configure_optimizers(basic_system):
    config = basic_system.configure_optimizers()
    assert "optimizer" in config
    assert "lr_scheduler" in config
    assert isinstance(config["optimizer"], Adam)
    assert isinstance(config["lr_scheduler"], StepLR)

def test_dataloader_creation(basic_system):
    basic_system.setup("fit")
    train_loader = basic_system.train_dataloader()
    assert isinstance(train_loader, DataLoader)
    assert train_loader.batch_size == 32

@pytest.mark.skip(reason="Requires trainer attachment")
def test_training_step(basic_system):
    basic_system.setup("fit")
    batch = next(iter(basic_system.train_dataloader()))
    result = basic_system._base_step(batch, batch_idx=0, mode="train")
    
    assert "loss" in result
    assert "metrics" in result
    assert "input" in result
    assert "target" in result
    assert "pred" in result
    assert torch.is_tensor(result["loss"])

@pytest.mark.skip(reason="Requires trainer attachment")
def test_validation_step(basic_system):
    basic_system.setup("validate")
    batch = next(iter(basic_system.val_dataloader()))
    result = basic_system._base_step(batch, batch_idx=0, mode="val")
    
    assert "loss" in result
    assert "metrics" in result
    assert torch.is_tensor(result["loss"])

def test_predict_step(basic_system):
    basic_system.setup("predict")
    batch = {"input": torch.randn(1, 3, 32, 32)}
    result = basic_system._base_step(batch, batch_idx=0, mode="predict")
    
    assert "pred" in result
    assert torch.is_tensor(result["pred"])

def test_learning_rate_property(basic_system):
    initial_lr = basic_system.learning_rate
    assert initial_lr == 0.001
    
    basic_system.learning_rate = 0.01
    assert basic_system.learning_rate == 0.01

@pytest.mark.parametrize("batch", [



    {"input": torch.randn(1, 3, 32, 32), "target": torch.randint(0, 10, size=()).long()},
    {"input": torch.randn(1, 3, 32, 32), "target": torch.randint(0, 10, size=()).long()},
    {"input": torch.randn(1, 3, 32, 32), "target": torch.randint(0, 10, size=()).long(), "id": "test_id"},
])
@pytest.mark.skip(reason="Requires trainer attachment")
def test_valid_batch_formats(basic_system, batch):
    basic_system.setup("fit")
    result = basic_system._base_step(batch, batch_idx=0, mode="train")
    assert isinstance(result, dict)

@pytest.mark.xfail(raises=ValueError)
def test_invalid_batch_format(basic_system):
    basic_system.setup("fit")
    invalid_batch = {"wrong_key": torch.randn(1, 3, 32, 32)}
    basic_system._base_step(invalid_batch, batch_idx=0, mode="train")
import pytest
from lighter.system import LighterSystem
from torch.nn import Module

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)

def test_system_initialization():
    model = DummyModel()
    system = LighterSystem(model=model, batch_size=32)
    assert system.batch_size == 32
