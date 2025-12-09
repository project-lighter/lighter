"""
LighterDataModule - A simple wrapper for organizing dataloaders in YAML configs.

This module provides LighterDataModule, a helper class that wraps PyTorch dataloaders
so they can be configured in YAML without requiring a custom LightningDataModule.
"""

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class LighterDataModule(LightningDataModule):
    """
    A lightweight wrapper for organizing dataloaders in configuration files.

    This class exists purely as a convenience helper - it wraps pre-configured
    PyTorch DataLoaders so you can use Lighter's configuration system without
    having to write a custom LightningDataModule from scratch.

    When to use LighterDataModule:
    - Simple datasets that don't need complex preprocessing
    - Quick experiments where you want to configure dataloaders in YAML
    - Cases where your data pipeline is straightforward

    When to write a custom LightningDataModule:
    - Complex data preparation (downloading, extraction, processing)
    - Multi-process data setup with prepare_data() and setup()
    - Advanced preprocessing pipelines
    - Data that requires stage-specific transformations
    - Sharing reusable data modules across projects

    Args:
        train_dataloader: DataLoader for training (used in fit stage)
        val_dataloader: DataLoader for validation (used in fit and validate stages)
        test_dataloader: DataLoader for testing (used in test stage)
        predict_dataloader: DataLoader for predictions (used in predict stage)

    Example:
        ```yaml
        # config.yaml
        data:
          _target_: lighter.LighterDataModule
          train_dataloader:
            _target_: torch.utils.data.DataLoader
            batch_size: 32
            shuffle: true
            dataset:
              _target_: torchvision.datasets.CIFAR10
              root: ./data
              train: true
              transform:
                _target_: torchvision.transforms.ToTensor
          val_dataloader:
            _target_: torch.utils.data.DataLoader
            batch_size: 32
            shuffle: false
            dataset:
              _target_: torchvision.datasets.CIFAR10
              root: ./data
              train: false
              transform:
                _target_: torchvision.transforms.ToTensor

        model:
          _target_: project.MyModel
          network: ...
          optimizer: ...

        trainer:
          _target_: pytorch_lightning.Trainer
          max_epochs: 10
        ```

    Note:
        This is just a thin wrapper around PyTorch Lightning's LightningDataModule.
        It doesn't add any special logic - it simply holds your dataloaders and
        returns them when Lightning asks for them.

        If you need more control (prepare_data, setup, etc.), write a custom
        LightningDataModule instead.
    """

    def __init__(
        self,
        train_dataloader: DataLoader | None = None,
        val_dataloader: DataLoader | None = None,
        test_dataloader: DataLoader | None = None,
        predict_dataloader: DataLoader | None = None,
    ) -> None:
        super().__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
        self._predict_dataloader = predict_dataloader

    def train_dataloader(self) -> DataLoader | None:
        """Return the training dataloader."""
        return self._train_dataloader

    def val_dataloader(self) -> DataLoader | None:
        """Return the validation dataloader."""
        return self._val_dataloader

    def test_dataloader(self) -> DataLoader | None:
        """Return the test dataloader."""
        return self._test_dataloader

    def predict_dataloader(self) -> DataLoader | None:
        """Return the prediction dataloader."""
        return self._predict_dataloader
