from typing import Any, Callable, Dict, Union

import gc
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger
from pytorch_lightning import Callback, Trainer

from lighter import LighterSystem


class LighterBaseWriter(ABC, Callback):
    """
    Base class for defining custom Writer. It provides the structure to save predictions in various formats.

    Subclasses should implement:
        1) `self.writers` attribute to specify the supported formats and their corresponding writer functions.
        2) `self.write()` method to specify the saving strategy for a prediction.

    Args:
        path (Union[str, Path]): Path for saving. It can be a directory or a specific file.
        writer (Union[str, Callable]): Name of the writer function registered in `self.writers`, or a custom writer function.
    """

    def __init__(self, path: Union[str, Path], writer: Union[str, Callable]) -> None:
        self.path = Path(path)

        # Check if the writer is a string and if it exists in the writers dictionary
        if isinstance(writer, str):
            if writer not in self.writers:
                raise ValueError(f"Writer for format {writer} does not exist. Available writers: {self.writers.keys()}.")
            self.writer = self.writers[writer]
        else:
            # If the writer is not a string, it is assumed to be a callable function
            self.writer = writer

        # Prediction counter. Used when IDs are not provided. Initialized in `self.setup()` based on the DDP rank.
        self._pred_counter = None

    @property
    @abstractmethod
    def writers(self) -> Dict[str, Callable]:
        """
        Property to define the default writer functions.
        """

    @abstractmethod
    def write(self, tensor: torch.Tensor, id: int) -> None:
        """
        Method to define how a tensor should be saved. The input tensor will be a single tensor without
        the batch dimension.

        For each supported format, there should be a corresponding writer function registered in `self.writers`
        A specific writer function can be retrieved using `self.get_writer(self.format)`.

        Args:
            tensor (torch.Tensor): Tensor, without the batch dimension, to be saved.
            id (int): Identifier for the tensor, can be used for naming files or adding table records.
        """

    def setup(self, trainer: Trainer, pl_module: LighterSystem, stage: str) -> None:
        """
        Callback function to set up necessary prerequisites: prediction count and prediction file or directory.
        When executing in a distributed environment, it ensures that:
        1. Each distributed node initializes a prediction count based on its rank.
        2. All distributed nodes write predictions to the same path.
        3. The path is accessible to all nodes, i.e., all nodes share the same storage.
        """
        if stage != "predict":
            return

        # Initialize the prediction count with the rank of the current process
        self._pred_counter = torch.distributed.get_rank() if trainer.world_size > 1 else 0

        # Ensure all distributed nodes write to the same path
        self.path = trainer.strategy.broadcast(self.path, src=0)
        directory = self.path.parent if self.path.suffix else self.path

        # Warn if the path already exists
        if self.path.exists():
            logger.warning(f"{self.path} already exists, existing predictions will be overwritten.")

        if trainer.is_global_zero:
            directory.mkdir(parents=True, exist_ok=True)

        # Wait for rank 0 to create the directory
        trainer.strategy.barrier()

        # Ensure all distributed nodes have access to the path
        if not directory.exists():
            raise RuntimeError(
                f"Rank {trainer.global_rank} does not share storage with rank 0. Ensure nodes have common storage access."
            )

    def on_predict_batch_end(
        self, trainer: Trainer, pl_module: LighterSystem, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """
        Callback method executed at the end of each prediction batch/step.
        If the IDs are not provided, it generates global unique IDs based on the prediction count.
        Finally, it writes the predictions using the specified writer.
        """
        # If the IDs are not provided, generate global unique IDs based on the prediction count. DDP supported.
        if outputs["id"] is None:
            batch_size = len(outputs["pred"])
            world_size = trainer.world_size
            outputs["id"] = list(range(self._pred_counter, self._pred_counter + batch_size * world_size, world_size))
            self._pred_counter += batch_size * world_size

        for id, pred in zip(outputs["id"], outputs["pred"]):
            self.write(tensor=pred, id=id)

        # Clear the predictions to save CPU memory. https://github.com/Lightning-AI/pytorch-lightning/issues/19398
        trainer.predict_loop._predictions = [[] for _ in range(trainer.predict_loop.num_dataloaders)]
        gc.collect()
