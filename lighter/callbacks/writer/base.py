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
from lighter.callbacks.utils import parse_data, structure_preserving_concatenate


class LighterBaseWriter(ABC, Callback):
    """Base class for a Writer. Override `self.write()` to define how a prediction should be saved.
    `LighterBaseWriter` sets up the write directory, and defines `on_predict_batch_end` and
    `on_predict_epoch_end`. `write_interval` specifies which of the two should the writer call.

    Args:
        write_dir (str): the Writer will create a directory inside of `write_dir` with date
            and time as its name and store the predictions there.
        write_as (Optional[Union[str, List[str], Dict[str, str], Dict[str, List[str]]]]):
            type in which the predictions will be stored. Passed automatically to the `write()`
            abstract method and can be used to support writing different types. Should the Writer
            support only one type, this argument can be removed from the overriden `__init__()`'s
            arguments and set `self.write_as = None`.
        write_interval (str, optional): whether to write on each step or at the end of the prediction epoch.
            Defaults to "step".
    """

    def __init__(
        self,
        write_dir: str,
        write_as: Optional[Union[str, List[str], Dict[str, str], Dict[str, List[str]]]],
        write_interval: str = "step",
    ) -> None:
        self.write_dir = Path(write_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.write_as = write_as
        self.write_interval = write_interval

        self.parsed_write_as = None

    @abstractmethod
    def write(
        self,
        idx: int,
        identifier: Optional[str],
        tensor: torch.Tensor,
        write_as: Optional[Union[str, List[str], Dict[str, str], Dict[str, List[str]]]],
    ):
        """This method must be overridden to specify how a tensor should be saved. If the Writer
        supports multiple types of saving, handle the `write_as` argument with an if-else statement.

        If the Writer only supports one type, remove `write_as` from the overridden
        `__init__()` method and set `self.write_as=None`.

        The `idx` and `identifier` arguments can be used to specify the name of the file
        or the row and column of a table for the prediction.

        Parameters:
            idx (int): The index of the prediction.
            identifier (Optional[str]): The identifier of the prediction. It will be `None` if there's
                only one prediction, an index if the prediction is a list of predictions, a key if it's
                a dict of predictions, and a key_index if it's a dict of list of predictions.
            tensor (torch.Tensor): The predicted tensor.
            write_as (Optional[Union[str, List[str], Dict[str, str], Dict[str, List[str]]]]):
                Specifies how to write the predictions. If it's a single string value, the predictions
                will be saved under that type regardless of whether they are single- or multi-output
                predictions. To write different outputs in the multi-output predictions using different
                methods, use the appropriate format for `write_as`.
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
        # Concatenate/flatten so that each output corresponds to its index.
        indices = list(itertools.chain(*indices))
        outputs = structure_preserving_concatenate(outputs)
        self._on_batch_or_epoch_end(outputs, indices)

    def _on_batch_or_epoch_end(self, outputs, indices):
        # Parse the outputs into a structure ready for writing.
        parsed_outputs = parse_data(outputs)
        # Runs only on the first step.
        if self.parsed_write_as is None:
            # Parse `self.write_as`. If multi-value, check if its structure matches `parsed_output`'s  structure.
            self.parsed_write_as = self._parse_write_as(self.write_as, parsed_outputs)

        for idx in indices:
            for identifier in parsed_outputs:  # pylint: disable=consider-using-dict-items
                tensor = parsed_outputs[identifier]
                write_as = self.parsed_write_as[identifier]
                self.write(idx, identifier, tensor, write_as)

    def _parse_write_as(self, write_as, parsed_outputs: Dict[str, Any]):
        # If `write_as` is a string (single value), all outputs will be saved in that specified format.
        if isinstance(write_as, str):
            parsed_write_as = {key: write_as for key in parsed_outputs}
        # Otherwise, `write_as` needs to match the structure of the outputs in order to assign each tensor its type.
        else:
            parsed_write_as = parse_data(write_as)
            if not set(parsed_write_as) == set(parsed_outputs):
                logger.error("`write_as` structure does not match the prediction's structure.")
                sys.exit()
        return parsed_write_as
