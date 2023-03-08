from typing import Any, Dict, List, Optional, Union

import itertools
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger
from pytorch_lightning import Callback, Trainer

from lighter import LighterSystem
from lighter.callbacks.utils import gather_tensors, parse_data


class LighterBaseWriter(ABC, Callback):
    """Base class for a Writer. Override `self.write()` to define how a prediction should be saved.
    `LighterBaseWriter` sets up the write directory, and defines `on_predict_batch_end` and
    `on_predict_epoch_end`. `write_interval` specifies which of the two should the writer call.

    Args:
        write_dir (str): the Writer will create a directory inside of `write_dir` with date
            and time as its name and store the predictions there.
        write_format (Optional[Union[str, List[str], Dict[str, str], Dict[str, List[str]]]]):
            type in which the predictions will be stored. Passed automatically to the `write()`
            abstract method and can be used to support writing different types. Should the Writer
            support only one type, this argument can be removed from the overriden `__init__()`'s
            arguments and set `self.write_format = None`.
        write_interval (str, optional): whether to write on each step or at the end of the prediction epoch.
            Defaults to "step".
    """

    def __init__(
        self,
        write_dir: str,
        write_format: Optional[Union[str, List[str], Dict[str, str], Dict[str, List[str]]]],
        write_interval: str = "step",
    ) -> None:
        self.write_dir = Path(write_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.write_format = write_format
        self.write_interval = write_interval

        self.parsed_write_format = None

    @abstractmethod
    def write(
        self,
        idx: int,
        identifier: Optional[str],
        tensor: torch.Tensor,
        write_format: Optional[Union[str, List[str], Dict[str, str], Dict[str, List[str]]]],
    ):
        """This method must be overridden to specify how a tensor should be saved. If the Writer
        supports multiple types of saving, handle the `write_format` argument with an if-else statement.

        If the Writer only supports one type, remove `write_format` from the overridden
        `__init__()` method and set `self.write_format=None`.

        The `idx` and `identifier` arguments can be used to specify the name of the file
        or the row and column of a table for the prediction.

        Parameters:
            idx (int): The index of the prediction.
            identifier (Optional[str]): The identifier of the prediction. It will be `None` if there's
                only one prediction, an index if the prediction is a list of predictions, a key if it's
                a dict of predictions, and a key_index if it's a dict of list of predictions.
            tensor (torch.Tensor): The predicted tensor.
            write_format (Optional[Union[str, List[str], Dict[str, str], Dict[str, List[str]]]]):
                Specifies how to write the predictions. If it's a single string value, the predictions
                will be saved under that type regardless of whether they are single- or multi-output
                predictions. To write different outputs in the multi-output predictions using different
                methods, use the appropriate format for `write_format`.
        """

    def setup(self, trainer: Trainer, pl_module: LighterSystem, stage: str) -> None:
        if stage != "predict":
            return

        if self.write_interval not in ["step", "epoch"]:
            logger.error("`write_interval` must be either 'step' or 'epoch'.")
            sys.exit()

        # Broadcast the `write_dir` so that all ranks write their predictions there.
        self.write_dir = trainer.strategy.broadcast(self.write_dir)
        # Let rank 0 create the `write_dir`.
        if trainer.is_global_zero:
            self.write_dir.mkdir(parents=True)
        # If `write_dir` does not exist, the ranks are not on the same storage.
        if not self.write_dir.exists():
            logger.error(
                f"Rank {trainer.global_rank} is not on the same storage as rank 0."
                "Please run the prediction only on nodes that are on the same storage."
            )
            sys.exit()

    def on_predict_batch_end(
        self, trainer: Trainer, pl_module: LighterSystem, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if self.write_interval != "step":
            return
        indices = trainer.predict_loop.epoch_loop.current_batch_indices
        self._on_batch_or_epoch_end(outputs, indices)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LighterSystem, outputs: List[Any]) -> None:
        if self.write_interval != "epoch":
            return
        # Only one epoch when predicting, index the lists of outputs and batch indices accordingly.
        indices = trainer.predict_loop.epoch_batch_indices[0]
        outputs = outputs[0]
        # Flatten so that each output sample corresponds to its index.
        indices = list(itertools.chain(*indices))
        # Remove the batch dimension since every output is a single sample.
        outputs = [output.squeeze(0) for output in outputs]
        # Gather the output tensors from all samples into a single structure rather than having one structures for each sample.
        outputs = gather_tensors(outputs)
        self._on_batch_or_epoch_end(outputs, indices)

    def _on_batch_or_epoch_end(self, outputs, indices):
        """Iterate through each output and save it in the specified format. The outputs and indices are automatically
        split individually by PyTorch Lightning."""
        # Sanity check. Should never happen. If it does, something is wrong with the Trainer.
        assert len(indices) == len(outputs)
        # `idx` is the index of the input sample, `output` is the output of the model for that sample.
        for idx, output in zip(indices, outputs):
            # Parse the outputs into a structure ready for writing.
            parsed_output = parse_data(output)
            # Parse `self.write_format`. If multi-value, check if its structure matches `parsed_output`'s structure.
            if self.parsed_write_format is None:
                self.parsed_write_format = self._parse_write_format(self.write_format, parsed_output)
            # Iterate through each prediction for the `idx`-th input sample.
            for identifier, tensor in parsed_output.items():
                # Save the prediction in the specified format.
                self.write(idx, identifier, tensor.detach().cpu(), self.parsed_write_format[identifier])

    def _parse_write_format(self, write_format, parsed_outputs: Dict[str, Any]):
        # If `write_format` is a string (single value), all outputs will be saved in that specified format.
        if isinstance(write_format, str):
            parsed_write_format = {key: write_format for key in parsed_outputs}
        # Otherwise, `write_format` needs to match the structure of the outputs in order to assign each tensor its type.
        else:
            parsed_write_format = parse_data(write_format)
            if not set(parsed_write_format) == set(parsed_outputs):
                logger.error("`write_format` structure does not match the prediction's structure.")
                sys.exit()
        return parsed_write_format
