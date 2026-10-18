"""LighterDataModule: dataloaders configured in YAML."""

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class LighterDataModule(LightningDataModule):
    """
    Holds ready-made dataloaders so they can be configured in YAML without writing a LightningDataModule.

    Write your own LightningDataModule instead when data needs `prepare_data()` (downloads) or
    stage-dependent `setup()`.

    Args:
        train_dataloader: DataLoader for training (used in fit stage)
        val_dataloader: DataLoader for validation (used in fit and validate stages)
        test_dataloader: DataLoader for testing (used in test stage)
        predict_dataloader: DataLoader for predictions (used in predict stage)

    Example:
        ```yaml
        data:
          _target_: lighter.LighterDataModule
          train_dataloader:
            _target_: torch.utils.data.DataLoader
            batch_size: 32
            shuffle: true
            dataset:
              _target_: project.data.MyDataset
        ```
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
