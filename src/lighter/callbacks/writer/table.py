"""
This module provides the TableWriter class, which saves predictions in a table format, such as CSV.
"""

import csv
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pytorch_lightning import Trainer

from lighter import System
from lighter.callbacks.writer.base import BaseWriter


class TableWriter(BaseWriter):
    """
    Writer for saving predictions in a table format, such as CSV.

    This implementation is designed to be scalable, especially for distributed prediction. Each process
    writes its predictions to a temporary file, and at the end of the epoch, the rank 0 process
    merges these files into a single output file.

    Args:
        path: Path to the output CSV file.
        writer: Writer function or a registered writer name for custom tensor serialization.
    """

    def __init__(self, path: str | Path, writer: str | Callable) -> None:
        super().__init__(path, writer)
        self._temp_path = None
        self._file_handle = None
        self._csv_writer = None

    @property
    def writers(self) -> dict[str, Callable]:
        """
        Returns a dictionary of default writer functions for tensors.
        'tensor' will convert a tensor to a list or a scalar, depending on its size.
        """
        return {
            "tensor": lambda tensor: tensor.item() if tensor.numel() == 1 else tensor.tolist(),
        }

    def setup(self, trainer: Trainer, pl_module: System, stage: str) -> None:
        """
        Set up the writer. For distributed training, a temporary file is created for each process to avoid
        gathering all data on rank 0.
        """
        super().setup(trainer, pl_module, stage)
        if stage != "predict":
            return

        # For distributed scenarios, each process writes to a temporary file.
        if trainer.world_size > 1:
            self._temp_path = self.path.parent / f"_temp_{trainer.global_rank}.csv"
        else:
            self._temp_path = self.path

        # Create a file handle and a CSV writer.
        self._file_handle = open(self._temp_path, "w", newline="")
        self._csv_writer = csv.writer(self._file_handle)
        self._csv_writer.writerow(["identifier", "pred"])

    def write(self, tensor: Any, identifier: int | str) -> None:
        """
        Writes a tensor as a record to the temporary CSV file.

        Args:
            tensor: The tensor to be written, expected to not have a batch dimension.
            identifier: A unique identifier for the record.
        """
        record = [identifier, self.writer(tensor)]
        self._csv_writer.writerow(record)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: System) -> None:
        """
        At the end of the prediction epoch, close the file handles. In a distributed setting, rank 0 is responsible
        for merging the temporary files into a single, sorted CSV file and then deleting the temporary files.
        """
        # Close the file handle.
        if self._file_handle:
            self._file_handle.close()

        # In a distributed setting, wait for all processes to finish writing.
        if trainer.world_size > 1:
            trainer.strategy.barrier()

        # Let rank 0 merge the files.
        if trainer.is_global_zero:
            if trainer.world_size > 1:
                # Find all temporary files.
                temp_files = list(self.path.parent.glob("_temp_*.csv"))
                # Read and concatenate them into a single DataFrame.
                df = pd.concat([pd.read_csv(f) for f in temp_files], ignore_index=True)
                # Delete the temporary files.
                for f in temp_files:
                    f.unlink()
            else:
                df = pd.read_csv(self.path)

            # Sort, set index, and save the final CSV file.
            try:
                df = df.sort_values("identifier")
            except TypeError:
                # The identifier might not be sortable.
                pass
            df = df.set_index("identifier")
            df.to_csv(self.path)

        # Reset state.
        self._temp_path = None
        self._file_handle = None
        self._csv_writer = None
