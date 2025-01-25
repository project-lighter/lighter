"""
This module provides the base class for defining custom writers in Lighter,
allowing predictions to be saved in various formats.
"""

from typing import Any, Callable

import gc
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from loguru import logger
from pytorch_lightning import Callback, Trainer
from torch import Tensor

from lighter import System
from lighter.utils.types.enums import Data, Stage


class BaseWriter(ABC, Callback):
    """
    Base class for defining custom Writer. It provides the structure to save predictions in various formats.

    Subclasses should implement:
        1) `self.writers` attribute to specify the supported formats and their corresponding writer functions.
        2) `self.write()` method to specify the saving strategy for a prediction.

    Args:
        path (str | Path): Path for saving predictions.
        writer (str | Callable): Writer function or name of a registered writer.
    """

    def __init__(self, path: str | Path, writer: str | Callable) -> None:
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
    def writers(self) -> dict[str, Callable]:
        """
        Property to define the default writer functions.
        """

    @abstractmethod
    def write(self, tensor: Tensor, identifier: int) -> None:
        """
        Method to define how a tensor should be saved. The input tensor will be a single tensor without
        the batch dimension.

        For each supported format, there should be a corresponding writer function registered in `self.writers`
        A specific writer function can be retrieved using `self.get_writer(self.format)`.

        Args:
            tensor (Tensor): Tensor, without the batch dimension, to be saved.
            identifier (int): Identifier for the tensor, can be used for naming files or adding table records.
        """

    def setup(self, trainer: Trainer, pl_module: System, stage: str) -> None:
        """
        Sets up the writer, ensuring the path is ready for saving predictions.

        Args:
            trainer (Trainer): The trainer instance.
            pl_module (System): The System instance.
            stage (str): The current stage of training.
        """
        if stage != Stage.PREDICT:
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
        self, trainer: Trainer, pl_module: System, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """
        Callback method executed at the end of each prediction batch to write predictions with unique IDs.

        Args:
            trainer (Trainer): The trainer instance.
            pl_module (System): The System instance.
            outputs (Any): The outputs from the prediction step.
            batch (Any): The current batch.
            batch_idx (int): The index of the batch.
            dataloader_idx (int): The index of the dataloader.
        """
        # If the IDs are not provided, generate global unique IDs based on the prediction count. DDP supported.
        if outputs[Data.IDENTIFIER] is None:
            batch_size = len(outputs[Data.PRED])
            world_size = trainer.world_size
            outputs[Data.IDENTIFIER] = list(
                range(
                    self._pred_counter,  # Start: counted globally, initialized with the rank of the current process
                    self._pred_counter + batch_size * world_size,  # Stop: count the total batch size across all processes
                    world_size,  # Step: each process writes predictions for every Nth sample
                )
            )
            self._pred_counter += batch_size * world_size

        for pred, identifier in zip(outputs[Data.PRED], outputs[Data.IDENTIFIER]):
            self.write(tensor=pred, identifier=identifier)

        # Clear the predictions to save CPU memory. https://github.com/Lightning-AI/pytorch-lightning/issues/19398
        # pylint: disable=protected-access
        trainer.predict_loop._predictions = [[] for _ in range(trainer.predict_loop.num_dataloaders)]
        gc.collect()
