"""Tests for LighterDataModule."""

import pickle

import pytest
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch.utils.data import DataLoader, TensorDataset

from lighter import LighterDataModule


class TestLighterDataModule:
    """Test suite for LighterDataModule."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a simple tensor dataset for testing."""
        x = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        return TensorDataset(x, y)

    @pytest.fixture
    def train_dataloader(self, sample_dataset):
        """Create a training dataloader."""
        return DataLoader(sample_dataset, batch_size=32, shuffle=True)

    @pytest.fixture
    def val_dataloader(self, sample_dataset):
        """Create a validation dataloader."""
        return DataLoader(sample_dataset, batch_size=32, shuffle=False)

    def test_initialization_all_dataloaders(self, train_dataloader, val_dataloader):
        """Test initialization with all dataloaders."""
        datamodule = LighterDataModule(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=val_dataloader,
            predict_dataloader=val_dataloader,
        )

        assert datamodule.train_dataloader() is train_dataloader
        assert datamodule.val_dataloader() is val_dataloader
        assert datamodule.test_dataloader() is val_dataloader
        assert datamodule.predict_dataloader() is val_dataloader

    def test_initialization_partial_dataloaders(self, train_dataloader):
        """Test initialization with only some dataloaders."""
        datamodule = LighterDataModule(train_dataloader=train_dataloader)

        assert datamodule.train_dataloader() is train_dataloader
        for name in ("val_dataloader", "test_dataloader", "predict_dataloader"):
            assert not is_overridden(name, datamodule)
            with pytest.raises(MisconfigurationException, match=name):
                getattr(datamodule, name)()

    def test_initialization_no_dataloaders(self):
        """Test initialization with no dataloaders."""
        datamodule = LighterDataModule()

        for name in ("train_dataloader", "val_dataloader", "test_dataloader", "predict_dataloader"):
            assert not is_overridden(name, datamodule)
            with pytest.raises(MisconfigurationException, match=name):
                getattr(datamodule, name)()

    def test_is_lightning_datamodule(self):
        """Test that LighterDataModule is a LightningDataModule."""
        from pytorch_lightning import LightningDataModule

        datamodule = LighterDataModule()
        assert isinstance(datamodule, LightningDataModule)

    def test_dataloader_returns_original_instance(self, train_dataloader):
        """Test that dataloaders return the exact instance passed in."""
        datamodule = LighterDataModule(train_dataloader=train_dataloader)

        # Should be the same instance, not a copy
        assert datamodule.train_dataloader() is train_dataloader

    def test_batch_iteration(self, train_dataloader):
        """Test that we can iterate over batches from the datamodule."""
        datamodule = LighterDataModule(train_dataloader=train_dataloader)

        # Should be able to iterate
        dataloader = datamodule.train_dataloader()
        batch = next(iter(dataloader))

        assert len(batch) == 2  # x, y
        assert batch[0].shape[0] == 32  # batch size
        assert batch[0].shape[1] == 10  # input dim


def test_absent_loader_hooks_preserve_native_identity_after_pickle():
    datamodule = pickle.loads(pickle.dumps(LighterDataModule()))
    for name in ("train_dataloader", "val_dataloader", "test_dataloader", "predict_dataloader"):
        assert not is_overridden(name, datamodule)
        with pytest.raises(MisconfigurationException, match=name):
            getattr(datamodule, name)()
