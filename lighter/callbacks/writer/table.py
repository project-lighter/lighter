"""
This module provides the TableWriter class, which saves predictions in a table format, such as CSV.
"""

from typing import Any, Callable

import itertools
from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import Trainer

from lighter import System
from lighter.callbacks.writer.base import BaseWriter


class TableWriter(BaseWriter):
    """
    Writer for saving predictions in a table format, such as CSV.

    Args:
        path: CSV filepath.
        writer: Writer function or name of a registered writer.
    """

    def __init__(self, path: str | Path, writer: str | Callable) -> None:
        super().__init__(path, writer)
        self.csv_records = []

    @property
    def writers(self) -> dict[str, Callable]:
        return {
            "tensor": lambda tensor: tensor.item() if tensor.numel() == 1 else tensor.tolist(),
        }

    def write(self, tensor: Any, identifier: int | str) -> None:
        """
        Writes the tensor as a table record using the specified writer.

        Args:
            tensor: The tensor to record. Should not have a batch dimension.
            identifier: Identifier for the record.
        """
        self.csv_records.append({"identifier": identifier, "pred": self.writer(tensor)})

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: System) -> None:
        """
        Called at the end of the prediction epoch to save predictions to a CSV file.

        Args:
            trainer: The trainer instance.
            pl_module: The System instance.
        """
        # If in distributed data parallel mode, gather records from all processes to rank 0.
        if trainer.world_size > 1:
            gather_csv_records = [None] * trainer.world_size if trainer.is_global_zero else None
            torch.distributed.gather_object(self.csv_records, gather_csv_records, dst=0)
            if trainer.is_global_zero:
                self.csv_records = list(itertools.chain(*gather_csv_records))

        # Save the records to a CSV file
        if trainer.is_global_zero:
            df = pd.DataFrame(self.csv_records)
            try:
                df = df.sort_values("identifier")
            except TypeError:
                pass
            df = df.set_index("identifier")
            df.to_csv(self.path)

        # Clear the records after saving
        self.csv_records = []
