"""Plain PyTorch Lightning modules for testing Lighter compatibility.

These modules demonstrate that Lighter works with ANY PyTorch Lightning module,
not just lighter.System. Users have complete freedom to use plain Lightning.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lighter import LighterModule


class SimpleDataset(Dataset):
    """Minimal dataset for testing."""

    def __init__(self, size=32):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(10)
        y = torch.randint(0, 2, (1,)).item()
        return x, y


class PlainLightningModule(pl.LightningModule):
    """
    A plain PyTorch Lightning module with NO Lighter-specific code.

    This demonstrates that Lighter works with any LightningModule.
    """

    def __init__(self, input_size=10, hidden_size=20, output_size=2, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        return DataLoader(SimpleDataset(32), batch_size=8)

    def val_dataloader(self):
        return DataLoader(SimpleDataset(16), batch_size=8)


class LightningModuleWithDataloaders(pl.LightningModule):
    """Lightning module that defines dataloaders internally."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(10, 2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return DataLoader(SimpleDataset(32), batch_size=8)

    def val_dataloader(self):
        return DataLoader(SimpleDataset(16), batch_size=8)


class MyLighterModule(LighterModule):
    """Example Lighter module for testing mixed usage."""

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.cross_entropy(pred, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        x, y = batch
        return {"pred": self(x)}

    def predict_step(self, batch, batch_idx):
        return self(batch[0])
