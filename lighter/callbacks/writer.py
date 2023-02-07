from typing import Any, Dict, List, Optional, Union

import itertools
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torchvision
from loguru import logger
from pytorch_lightning import Callback, Trainer

from lighter import LighterSystem
from lighter.callbacks.utils import concatenate, parse_data, preprocess_image


class LighterBaseWriter(ABC, Callback):
    def __init__(
        self,
        write_dir: str,
        write_as: Optional[Union[str, List[str], Dict[str, str], Dict[str, List[str]]]],
        write_on: str = "step",
    ) -> None:
        self.write_dir = Path(write_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.write_as = write_as
        self.write_on = write_on

        self.parsed_write_as = None

    @abstractmethod
    def write(
        self,
        idx: int,
        identifier: Optional[str],
        tensor: torch.Tensor,
        write_as: Optional[Union[str, List[str], Dict[str, str], Dict[str, List[str]]]],
    ):
        """This method should be overridden to specify how a tensor should be saved. If the Writer
        supports multiple types of saving, handle the `write_as` argument with an if-else statement.

        If the Writer only supports one type, `write_as` can be ignored and `write_as=None` can be
        set in the overridden `__init__()` method.

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

        if self.write_on not in ["step", "epoch"]:
            logger.error("`write_on` must be either 'step' or 'epoch'.")
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
        if self.write_on != "step":
            return
        indices = trainer.predict_loop.epoch_loop.current_batch_indices
        self._on_batch_or_epoch_end(outputs, indices)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LighterSystem, outputs: List[Any]) -> None:
        if self.write_on != "epoch":
            return
        # Only one epoch when predicting, index the lists of outputs and batch indices accordingly.
        indices = trainer.predict_loop.epoch_batch_indices[0]
        outputs = outputs[0]
        # Concatenate/flatten so that each output corresponds to its index.
        indices = list(itertools.chain(*indices))
        outputs = concatenate(outputs)
        self._on_batch_or_epoch_end(outputs, indices)

    def _on_batch_or_epoch_end(self, outputs, indices):
        # Parse the outputs into a structure ready for writing.
        parsed_outputs = parse_data(outputs)
        # Parse `write_as`. If multi-value, check if its structure matches `parsed_output`'s  structure.
        parsed_write_as = self._parse_write_as(self.write_as, parsed_outputs)

        for idx in indices:
            for identifier in parsed_outputs:  # pylint: disable=consider-using-dict-items
                tensor = parsed_outputs[identifier]
                write_as = parsed_write_as[identifier]
                self.write(idx, identifier, tensor, write_as)

    def _parse_write_as(self, write_as, parsed_outputs: Dict[str, Any]):
        if self.parsed_write_as is None:
            # If `write_as` is a string (single value), all outputs will be saved in that specified format.
            if isinstance(write_as, str):
                self.parsed_write_as = {key: write_as for key in parsed_outputs}
            # Otherwise, `write_as` needs to match the structure of the outputs in order to assign each tensor its type.
            else:
                self.parsed_write_as = parse_data(write_as)
                if not set(self.parsed_write_as) == set(parsed_outputs):
                    logger.error("`write_as` structure does not match the prediction's structure.")
                    sys.exit()
        return self.parsed_write_as


class LighterFileWriter(LighterBaseWriter):
    def write(self, idx, identifier, tensor, write_as):
        filename = f"{write_as}" if identifier is None else f"{identifier}_{write_as}"
        write_dir = self.write_dir / str(idx)
        write_dir.mkdir()

        if write_as is None:
            pass
        elif write_as == "tensor":
            path = write_dir / f"{filename}.pt"
            torch.save(tensor, path)
        elif write_as == "image":
            path = write_dir / f"{filename}.png"
            torchvision.io.write_png(preprocess_image(tensor), path)
        elif write_as == "video":
            path = write_dir / f"{filename}.mp4"
            torchvision.io.write_video(path, tensor, fps=24)
        elif write_as == "scalar":
            raise NotImplementedError
        elif write_as == "audio":
            raise NotImplementedError
        else:
            logger.error(f"`write_as` '{write_as}' not supported.")
            sys.exit()


class LighterTableWriter(LighterBaseWriter):
    def __init__(self, write_dir: str, write_as: Union[str, List[str], Dict[str, str], Dict[str, List[str]]]) -> None:
        super().__init__(write_dir, write_as, write_on="epoch")
        self.csv_records = {}

    def write(self, idx, identifier, tensor, write_as):
        # Column name will be set to 'pred' if the identifier is None.
        column = "pred" if identifier is None else identifier

        if write_as is None:
            record = None
        elif write_as == "tensor":
            record = tensor.tolist()
        elif write_as == "scalar":
            raise NotImplementedError
        else:
            logger.error(f"`write_as` '{write_as}' not supported.")
            sys.exit()

        if idx not in self.csv_records:
            self.csv_records[idx] = {column: record}
        else:
            self.csv_records[idx][column] = record

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LighterSystem, outputs: List[Any]) -> None:
        super().on_predict_epoch_end(trainer, pl_module, outputs)

        csv_path = self.write_dir / "predictions.csv"
        logger.info(f"Saving the predictions to {csv_path}")

        # Sort the dict of dicts by key and turn it into a list of dicts.
        self.csv_records = [self.csv_records[key] for key in sorted(self.csv_records)]
        # Gather the records from all ranks when in DDP.
        if trainer.world_size > 1:
            # Since `all_gather` supports tensors only, mimic the behavior using `broadcast`.
            ddp_csv_records = [self.csv_records] * trainer.world_size
            for rank in range(trainer.world_size):
                # Broadcast the records from the current rank and save it at its designated position.
                ddp_csv_records[rank] = trainer.strategy.broadcast(ddp_csv_records[rank], src=rank)
            # Combine the records from all ranks. List of lists of dicts -> list of dicts.
            self.csv_records = list(itertools.chain(*ddp_csv_records))

        # Create a dataframe and save it.
        pd.DataFrame(self.csv_records).to_csv(csv_path)
