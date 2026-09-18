from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger
from pytorch_lightning import Callback, LightningModule, Trainer

from lighter.utils.types.enums import Stage


class BaseWriter(ABC, Callback):
    """
    Base class for defining custom Writers. It provides a structure to save predictions.

    Subclasses should implement the `write` method to define the saving strategy.
    Prediction retention is owned by Trainer.predict(return_predictions=...).
    Use return_predictions=False for streaming without retaining batch outputs.

    Args:
        path (str | Path): Path for saving predictions.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @abstractmethod
    def write(self, outputs: dict[str, Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """
        Abstract method to define how the outputs of a prediction batch should be saved.
        Args:
            outputs: The dictionary of outputs from the prediction step.
            batch: The current batch.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.
        """

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != Stage.PREDICT:
            return

        self.path = trainer.strategy.broadcast(self.path, src=0)
        directory = self.path.parent if self.path.suffix else self.path

        if self.path.exists():
            logger.warning(f"{self.path} already exists, existing predictions will be overwritten.")

        if trainer.is_global_zero:
            directory.mkdir(parents=True, exist_ok=True)

        trainer.strategy.barrier()

        if not directory.exists():
            raise RuntimeError(
                f"Rank {trainer.global_rank} does not share storage with rank 0. Ensure nodes have common storage access."
            )

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not outputs:
            return
        self.write(outputs, batch, batch_idx, dataloader_idx)
