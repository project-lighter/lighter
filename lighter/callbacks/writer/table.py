from typing import Any, Dict, List, Union

import itertools
import sys

import pandas as pd
from loguru import logger
from pytorch_lightning import Trainer

from lighter import LighterSystem
from lighter.callbacks.writer.base import LighterBaseWriter


class LighterTableWriter(LighterBaseWriter):
    def __init__(self, write_dir: str, write_format: Union[str, List[str], Dict[str, str], Dict[str, List[str]]]) -> None:
        super().__init__(write_dir, write_format, write_interval="epoch")
        self.csv_records = {}

    def write(self, idx, identifier, tensor, write_format):
        # Column name will be set to 'pred' if the identifier is None.
        column = "pred" if identifier is None else identifier

        if write_format is None:
            record = None
        elif write_format == "tensor":
            record = tensor.tolist()
        elif write_format == "scalar":
            raise NotImplementedError
        else:
            logger.error(f"`write_format` '{write_format}' not supported.")
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
