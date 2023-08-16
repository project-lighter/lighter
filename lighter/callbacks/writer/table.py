from typing import Any, Callable, Dict, List, Optional, Union

import itertools
from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import Trainer

from lighter import LighterSystem
from lighter.callbacks.writer.base import LighterBaseWriter


class LighterTableWriter(LighterBaseWriter):
    """
    Writer for saving predictions in a table format. Supports multiple formats, and
    additional custom formats can be added either through `additional_writers`
    argument at initialization, or by calling `add_writer` method after initialization.

    Args:
        directory (Path): The directory where the CSV will be saved.
        format (str): The format in which the data should be saved in the CSV.
        additional_writers (Optional[Dict[str, Callable]]): Additional custom writer functions.
    """

    def __init__(
        self, directory: Union[str, Path], format: str, additional_writers: Optional[Dict[str, Callable]] = None
    ) -> None:
        # Predefined writers for different formats.
        self._writers = {
            "tensor": write_tensor,
        }

        # Initialize the base class.
        super().__init__(directory, format, additional_writers)

        # Create a dictionary to hold CSV records for each ID. These are populated at each batch end
        # by `self.on_predict_batch_end` defined in the base class using the `write` method below.
        # Finally, the records are dumped to a CSV file at the end of the epoch by `self.on_predict_epoch_end`.
        self.csv_records = {}

    def write(self, tensor: Any, id: Union[int, str], multi_pred_id: Optional[Union[int, str]], format: str) -> None:
        """
        Write the tensor as a table record in the given format.
        If there are multiple predictions, there will be a separate column for each prediction, named after
        the corresponding `multi_pred_id`. If single prediction, there will be a single column named "pred".

        Args:
            tensor (Any): The tensor to be written.
            id (Union[int, str]): The primary identifier for naming.
            multi_pred_id (Optional[Union[int, str]]): The secondary identifier, used if there are multiple predictions.
            format (str): Format in which tensor should be written.
        """
        # Determine the column name based on the presence of multi_pred_id
        column = "pred" if multi_pred_id is None else multi_pred_id

        # Get the appropriate writer function for the given format
        writer = self.get_writer(format)

        # Convert the tensor to the desired format (e.g., list)
        record = writer(tensor)

        # Store the record in the csv_records dictionary under the specified ID and column
        if id not in self.csv_records:
            self.csv_records[id] = {column: record}
        else:
            self.csv_records[id][column] = record

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        """
        Callback invoked at the end of the prediction epoch to save predictions to a CSV file.

        This method is responsible for organizing prediction records and saving them as a CSV file.
        If training was done in a distributed setting, it gathers predictions from all processes
        and then saves them from the rank 0 process.
        """
        # Set the path where the CSV will be saved
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


def write_tensor(tensor: Any) -> List:
    return tensor.tolist()
