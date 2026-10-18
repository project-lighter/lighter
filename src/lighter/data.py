"""LighterDataModule: dataloaders configured in YAML."""

from typing import Any

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class LighterDataModule(LightningDataModule):
    """
    Holds ready-made dataloaders so they can be configured in YAML without writing a LightningDataModule.

    Write your own LightningDataModule instead when data needs `prepare_data()` (downloads) or
    stage-dependent `setup()`. Leave out the dataloaders you don't use: without `val_dataloader`,
    `fit` trains without validation.

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
        self._loaders = {
            "train": train_dataloader,
            "val": val_dataloader,
            "test": test_dataloader,
            "predict": predict_dataloader,
        }
        for name, loader in self._loaders.items():
            if loader is None:  # as if the method were not written: Lightning skips the loop or names what is missing
                setattr(self, f"{name}_dataloader", None)

    def train_dataloader(self) -> Any:
        return self._loaders["train"]

    def val_dataloader(self) -> Any:
        return self._loaders["val"]

    def test_dataloader(self) -> Any:
        return self._loaders["test"]

    def predict_dataloader(self) -> Any:
        return self._loaders["predict"]
