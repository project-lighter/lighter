from typing import Any, Callable, Dict, Union

import itertools
from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import Trainer

from lighter import LighterSystem
from lighter.callbacks.writer.base import LighterBaseWriter


class LighterTableWriter(LighterBaseWriter):
    """
    Writer for saving predictions in a table format.

    Args:
        directory (Path): The directory where the CSV will be saved.
        writer (Union[str, Callable]): Name of the writer function registered in `self.writers` or a custom writer function.
            Available writers: "tensor". A custom writer function must take a single argument: `tensor`, and return the record
            to be saved in the CSV file. The tensor will be a single tensor without the batch dimension.
    """

    def __init__(self, directory: Union[str, Path], writer: Union[str, Callable]) -> None:
        super().__init__(directory, writer)
        self.csv_records = {}

    @property
    def writers(self) -> Dict[str, Callable]:
        return {
            "tensor": lambda tensor: tensor.tolist(),
        }

    def write(self, tensor: Any, id: Union[int, str]) -> None:
        """
        Write the tensor as a table record using the specified writer.

        Args:
            tensor (Any): tensor, without the batch dimension, to be recorded.
            id (Union[int, str]): identifier, used as the key for the record.
        """
        column = "pred"
        record = self.writer(tensor)
        self.csv_records.setdefault(id, {})[column] = record

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        """
        Callback invoked at the end of the prediction epoch to save predictions to a CSV file.

        This method is responsible for organizing prediction records and saving them as a CSV file.
        If training was done in a distributed setting, it gathers predictions from all processes
        and then saves them from the rank 0 process.
        """
        csv_path = self.directory / "predictions.csv"

        # Sort the records by ID and convert the dictionary to a list
        self.csv_records = [self.csv_records[id] for id in sorted(self.csv_records)]

        # If in distributed data parallel mode, gather records from all processes to rank 0.
        if trainer.world_size > 1:
            # Create a list to hold the records from each process. Used on rank 0 only.
            gather_csv_records = [None] * trainer.world_size if trainer.is_global_zero else None
            # Each process sends its records to rank 0, which stores them in the `gather_csv_records`.
            torch.distributed.gather_object(self.csv_records, gather_csv_records, dst=0)
            # Concatenate the gathered records
            if trainer.is_global_zero:
                self.csv_records = list(itertools.chain(*gather_csv_records))

        # Save the records to a CSV file
        if trainer.is_global_zero:
            pd.DataFrame(self.csv_records).to_csv(csv_path)
