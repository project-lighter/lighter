import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger
from pytorch_lightning import Callback, Trainer

from lighter.model import LighterModule
from lighter.utils.types.enums import Stage


class BaseWriter(ABC, Callback):
    """
    Base class for defining custom Writers. It provides a structure to save predictions.

    Subclasses should implement the `write` method to define the saving strategy.

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

    def setup(self, trainer: Trainer, pl_module: LighterModule, stage: str) -> None:
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
        pl_module: LighterModule,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not outputs:
            return
        self.write(outputs, batch, batch_idx, dataloader_idx)

        # Clear the predictions to save CPU memory. This is a temporary workaround for a known issue in PyTorch
        # Lightning, where predictions can accumulate in memory. This line accesses a private attribute
        # `_predictions` of the `predict_loop`, which is a brittle dependency and may break in future
        # versions of Lightning. For more details, see: https://github.com/Lightning-AI/pytorch-lightning/issues/19398
        trainer.predict_loop._predictions = [[] for _ in range(trainer.predict_loop.num_dataloaders)]
        gc.collect()
