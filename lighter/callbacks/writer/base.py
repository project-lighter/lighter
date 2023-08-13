from typing import Any, Callable, Dict, List, Optional, Union

import itertools
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger
from pytorch_lightning import Callback, Trainer

from lighter import LighterSystem
from lighter.callbacks.utils import group_tensors, parse_data


class LighterBaseWriter(ABC, Callback):
    """
    Base class for defining custom Writer. It provides the structure to save predictions in various formats.

    Subclasses should implement:
        1) `self._writers` property to specify the supported formats and their corresponding writer functions.
        2) `self.write()` method to specify the saving strategy for a prediction.

    Args:
        directory (str): Base directory for saving. A new sub-directory with current date and time will be created inside.
        format (Optional[Union[str, List[str], Dict[str, str], Dict[str, List[str]]]]): Desired format(s) for saving predictions.
            The format will be passed to the `write` method.
        interval (str, optional): Specifies when to save predictions - at every step or at the end of epoch. Defaults to "step".
        additional_writers (Optional[Dict[str, Callable]]): Additional writer functions to be registered with the base writer.
    """

    def __init__(
        self,
        directory: str,
        format: Optional[Union[str, List[str], Dict[str, str], Dict[str, List[str]]]],
        interval: str,
        additional_writers: Optional[Dict[str, Callable]] = None,
    ) -> None:
        # Create a unique directory using the current date and time
        self.directory = Path(directory) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.format = format
        self.interval = interval

        # Placeholder for processed format for quicker access during writes
        self.parsed_format = None

        # Keeps track of last written prediction index for cases when ids aren't provided
        self.last_index = 0

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
        Method to define how a tensor should be saved.

        Depending on the specified format, this method should contain logic to handle the saving mechanism.

        Args:
            tensor (torch.Tensor): Tensor to be saved.
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

        # Validate the interval parameter
        if self.interval not in ["step", "epoch"]:
            raise ValueError("`interval` must be either 'step' or 'epoch'.")

        # Ensure all distributed nodes write to the same directory
        self.directory = trainer.strategy.broadcast(self.directory)
        if trainer.is_global_zero:
            self.directory.mkdir(parents=True)
        if not self.directory.exists():
            raise RuntimeError(
                f"Rank {trainer.global_rank} does not share storage with rank 0. Ensure nodes have common storage access."
            )

    def on_predict_batch_end(
        self, trainer: Trainer, pl_module: LighterSystem, outputs: Any, batch: Any, batch_idx: int = 0
    ) -> None:
        """Callback method triggered at the end of each prediction batch/step."""
        if self.interval != "step":
            return

        preds, ids = outputs["pred"], outputs["id"]

        # Generate IDs if not provided
        if ids is None:
            ids = list(range(self.last_index, self.last_index + len(preds)))
            self.last_index += len(preds)

        self._on_batch_or_epoch_end(preds, ids)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LighterSystem, outputs: List[Any]) -> None:
        """Callback method triggered at the end of the prediction epoch."""
        if self.interval != "epoch":
            return
        # Only one epoch when predicting, select its outputs.
        preds, ids = outputs["pred"][0], outputs["id"][0]
        # Remove the batch dimension since every pred is a single sample.
        preds = [pred.squeeze(0) for pred in preds]
        # Group predictions from all samples into a unified structure.
        preds = group_tensors(preds)
        # If no ids provided, assign default sequential ids based on the prediction order.
        if ids[0] is None:
            ids = list(range(len(preds)))
        self._on_batch_or_epoch_end(preds, ids)

    def _on_batch_or_epoch_end(self, preds, ids):
        """
        Process each prediction at the end of either a batch or epoch and save in the defined format.

        Args:
            preds: Predicted tensors.
            ids: Corresponding identifiers for the predictions.
        """
        # Sanity check to ensure matched lengths for predictions and ids.
        assert len(ids) == len(preds)
        for id, pred in zip(ids, preds):
            # Convert predictions into a structured format suitable for writing.
            parsed_pred = parse_data(pred)
            # If the format hasn't been parsed yet, do it now.
            if self.parsed_format is None:
                self.parsed_format = parse_format(self.format, parsed_pred)
            # If multiple outputs, parsed_pred will contain multiple keys. For a single output, key will be None.
            for multi_pred_id, tensor in parsed_pred.items():
                # Save the prediction as per the designated format.
                self.write(tensor.detach().cpu(), id, multi_pred_id, self.parsed_format[multi_pred_id])

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


def parse_format(format: str, parsed_preds: Dict[str, Any]) -> Dict[str, str]:
    """
    Parse the given format and align it with the structure of the predictions.

    If the format is a single string, all predictions will be saved in this format. If the format has a structure
    (like a dictionary), it needs to match the structure of the predictions.

    Args:
        format (str): The storage format for the predictions, either as a string or a structured format.
        parsed_preds (Dict[str, Any]): Dictionary of parsed prediction data.

    Returns:
        Dict[str, str]: Dictionary of parsed format data corresponding to the prediction structure.

    Raises:
        ValueError: If the structure of the format does not align with the prediction structure.
    """
    if isinstance(format, str):
        # Assign the single format to all prediction keys.
        parsed_format = {key: format for key in parsed_preds}
    else:
        # Ensure the structured format corresponds with the predictions' structure.
        parsed_format = parse_data(format)
        if not set(parsed_format) == set(parsed_preds):
            raise ValueError("`format` structure does not match the prediction's structure.")
    return parsed_format
