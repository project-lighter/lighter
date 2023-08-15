from typing import Any, Callable, Dict, List, Optional, Union

import itertools
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger
from monai.data.utils import decollate_batch
from pytorch_lightning import Callback, Trainer

from lighter import LighterSystem
from lighter.callbacks.utils import parse_data, parse_format


class LighterBaseWriter(ABC, Callback):
    """
    Base class for defining custom Writer. It provides the structure to save predictions in various formats.

    Subclasses should implement:
        1) `self._writers` attribute to specify the supported formats and their corresponding writer functions.
        2) `self.write()` method to specify the saving strategy for a prediction.

    Args:
        directory (str): Base directory for saving. A new sub-directory with current date and time will be created inside.
        format (Optional[Union[str, List[str], Dict[str, str], Dict[str, List[str]]]]):
            Desired format(s) for saving predictions. The format will be passed to the `write` method.
        additional_writers (Optional[Dict[str, Callable]]): Additional writer functions to be registered with the base writer.
    """

    def __init__(
        self,
        directory: str,
        format: Optional[Union[str, List[str], Dict[str, str], Dict[str, List[str]]]],
        additional_writers: Optional[Dict[str, Callable]] = None,
    ) -> None:
        # Create a unique directory using the current date and time
        self.directory = Path(directory) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.format = format

        # Placeholder for processed format for quicker access during writes
        self._parsed_format = None

        # When IDs are not provided, keep track of the global prediction count. Supports DDP.
        self._pred_count = None

        # Ensure that default writers are defined
        if not hasattr(self, "_writers"):
            raise NotImplementedError("Subclasses of LighterBaseWriter must implement the `_writers` property.")

        # Register any additional writers passed during initialization
        if additional_writers:
            for format, writer_function in additional_writers.items():
                self.add_writer(format, writer_function)

    @abstractmethod
    def write(
        self,
        tensor: torch.Tensor,
        id: int,
        multi_pred_id: Optional[str],
        format: Optional[Union[str, List[str], Dict[str, str], Dict[str, List[str]]]],
    ) -> None:
        """
        Method to define how a tensor should be saved. The input tensor will be a single tensor without
        the batch dimension. If the batch dimension is needed, apply `tensor.unsqueeze(0)` before saving,
        either in this method or in the particular writer function.

        For each supported format, there should be a corresponding writer function registered in `self._writers`,
        and can be retrieved using `self.get_writer(format)`.

        Args:
            tensor (torch.Tensor): Tensor to be saved. It will be a single tensor without the batch dimension.
            id (int): Identifier for the tensor, can be used for naming or indexing.
            multi_pred_id (Optional[str]): Used when there are multiple predictions for a single input.
                It can represent the index of a prediction, the key of a prediction in case of a dict,
                or combined key and index for a dict of lists.
            format (Optional[Union[str, List[str], Dict[str, str], Dict[str, List[str]]]]): Format for saving the tensor.
        """
        pass

    def setup(self, trainer: Trainer, pl_module: LighterSystem, stage: str) -> None:
        """Callback for setup stage in Pytorch Lightning Trainer."""
        if stage != "predict":
            return

        # Initialize the prediction count with the rank of the current process
        self._pred_count = torch.distributed.get_rank() if trainer.world_size > 1 else 0

        # Ensure all distributed nodes write to the same directory
        self.directory = trainer.strategy.broadcast(self.directory, src=0)
        if trainer.is_global_zero:
            self.directory.mkdir(parents=True)
        # Wait for rank 0 to create the directory
        trainer.strategy.barrier()

        # Ensure all distributed nodes have access to the directory
        if not self.directory.exists():
            raise RuntimeError(
                f"Rank {trainer.global_rank} does not share storage with rank 0. Ensure nodes have common storage access."
            )

    def on_predict_batch_end(
        self, trainer: Trainer, pl_module: LighterSystem, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Callback method triggered at the end of each prediction batch/step."""
        # Fetch and decollate preds.
        preds = decollate_batch(outputs["pred"], detach=True, pad=False)
        # Fetch and decollate IDs if provided.
        if outputs["id"] is not None:
            ids = decollate_batch(outputs["id"], detach=True, pad=False)
        # Generate IDs if not provided. An ID will be the global index of the prediction.
        else:
            ids = []
            for _ in range(len(preds)):
                # Append the current prediction count to the IDs list.
                ids.append(self._pred_count)
                # Increment the prediction count by the total number of DDP processes.
                # This ensures each process will generate unique IDs in the next batch.
                self._pred_count += trainer.world_size

        # Iterate over the predictions and save them.
        for id, pred in zip(ids, preds):
            # Convert predictions into a structured format suitable for writing.
            parsed_pred = parse_data(pred)
            # If the format hasn't been parsed yet, do it now.
            if self._parsed_format is None:
                self._parsed_format = parse_format(self.format, parsed_pred)
            # If multiple outputs, parsed_pred will contain multiple keys. For a single output, key will be None.
            for multi_pred_id, tensor in parsed_pred.items():
                # Save the prediction as per the designated format.
                self.write(tensor, id, multi_pred_id, format=self._parsed_format[multi_pred_id])

    def add_writer(self, format: str, writer_function: Callable) -> None:
        """
        Register a new writer function for a specified format.

        Args:
            format (str): Format type for which the writer is being registered.
            writer_function (Callable): Function to write data in the given format.

        Raises:
            ValueError: If a writer for the given format is already registered.
        """
        if format in self._writers:
            raise ValueError(f"Writer for format {format} already registered.")
        self._writers[format] = writer_function

    def get_writer(self, format: str) -> Callable:
        """
        Retrieve the registered writer function for a specified format.

        Args:
            format (str): Format for which the writer function is needed.

        Returns:
            Callable: Registered writer function for the given format.

        Raises:
            ValueError: If no writer is registered for the specified format.
        """
        if format not in self._writers:
            raise ValueError(f"Writer for format {format} not registered.")
        return self._writers[format]
