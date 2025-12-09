"""
This module provides the CsvWriter class, which saves predictions in a table format, such as CSV.
"""

import csv
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from pytorch_lightning import Trainer

from lighter.callbacks.base_writer import BaseWriter
from lighter.model import LighterModule
from lighter.utils.types.enums import Stage


class CsvWriter(BaseWriter):
    """
    Writer for saving predictions in a CSV format. It accumulates predictions in a temporary
    file and saves them to the final destination at the end of the prediction epoch.

    Args:
        path (str | Path): Path to save the final CSV file.
        keys (list[str]): A list of keys to be included in the CSV file.
                          These keys must be present in the `outputs` dictionary
                          from the prediction step.

    Example:
        ```yaml
        trainer:
          callbacks:
            - _target_: lighter.callbacks.CsvWriter
              path: predictions.csv
              keys: [id, pred, target]
        ```
    """

    def __init__(self, path: str | Path, keys: list[str]) -> None:
        super().__init__(path)
        self.keys = keys
        self._temp_path: Path | None = None
        self._csv_writer: Any = None  # csv.writer type is not easily annotated
        self._csv_file: TextIOWrapper | None = None

    def _close_file(self) -> None:
        """Close the CSV file if it's open and reset related state."""
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.close()
        self._csv_file = None
        self._csv_writer = None

    def setup(self, trainer: Trainer, pl_module: LighterModule, stage: str) -> None:
        if stage != Stage.PREDICT:
            return
        super().setup(trainer, pl_module, stage)

        # Create a temporary file for writing predictions
        self._temp_path = self.path.with_suffix(f".tmp_rank{trainer.global_rank}{self.path.suffix}")
        self._csv_file = open(self._temp_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        # Write header
        self._csv_writer.writerow(self.keys)

    def _get_sequence_length(self, value: Any) -> int | None:
        if isinstance(value, (list, tuple)):
            return len(value)
        elif isinstance(value, torch.Tensor):
            if value.ndim == 0:  # Scalar tensor
                return 1
            else:
                return len(value)  # For non-scalar tensors, len() works
        return None  # Not a sequence type we care about

    def _get_record_value(self, value: Any, index: int) -> Any:
        if isinstance(value, (list, tuple)):
            return value[index]
        elif isinstance(value, torch.Tensor):
            if value.ndim == 0:  # Scalar tensor
                return value.item()  # Get Python scalar
            else:
                # For non-scalar tensors, get the item at index.
                # If the item itself is a scalar tensor, convert to Python scalar.
                # Otherwise, convert to a list (e.g., for image data).
                item = value[index]
                return item.item() if item.ndim == 0 else item.tolist()
        else:
            return value  # Non-sequence value, return as is (assumed to be for all samples)

    def write(self, outputs: dict[str, Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if self._csv_writer is None:
            return

        # Validate that at least one configured key is present in outputs
        present_keys = [key for key in self.keys if key in outputs]
        if not present_keys:
            missing_keys = self.keys
            raise KeyError(
                f"CsvWriter: none of the configured keys {missing_keys} were found in outputs. "
                f"Available keys in outputs: {list(outputs.keys())}"
            )

        # Determine the number of samples in the batch.
        num_samples = 0
        for key in self.keys:
            if key in outputs:
                length = self._get_sequence_length(outputs[key])
                if length is not None:
                    num_samples = length
                    break
                else:
                    # If it's not a sequence type we handle, assume it's a single sample
                    if num_samples == 0:
                        num_samples = 1

        # Validate that all list-like or tensor outputs have the same length
        for key in self.keys:
            if key in outputs:
                current_len = self._get_sequence_length(outputs[key])

                # Only validate if it's a sequence type and its length is not None
                if current_len is not None and current_len != num_samples:
                    raise ValueError(
                        f"CsvWriter found inconsistent lengths for keys: "
                        f"expected {num_samples}, but found {current_len} for key '{key}'."
                    )

        # Transpose the dictionary of lists into a list of per-sample records and write to CSV
        for i in range(num_samples):
            record = []
            for key in self.keys:
                if key not in outputs:
                    raise KeyError(f"CsvWriter expected key '{key}' in outputs but it was missing.")

                value = outputs[key]
                record.append(self._get_record_value(value, i))
            self._csv_writer.writerow(record)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LighterModule) -> None:
        """
        At the end of the prediction epoch, it saves the temporary file to the final destination.
        """
        if self._csv_file is None:
            return

        # Close the temporary file
        self._close_file()

        all_temp_paths: list[Path | None] = [None] * trainer.world_size
        if dist.is_initialized():
            dist.all_gather_object(all_temp_paths, self._temp_path)
        else:
            all_temp_paths = [self._temp_path]

        if trainer.is_global_zero:
            # Read all temporary files into pandas DataFrames and concatenate them
            dfs = [pd.read_csv(path) for path in all_temp_paths if path is not None]
            if not dfs:
                return
            df = pd.concat(dfs, ignore_index=True)

            # Save the final CSV file
            df.to_csv(self.path, index=False)

            # Remove all temporary files
            for path in all_temp_paths:
                if path is not None:
                    path.unlink()

        # Reset temporary path
        self._temp_path = None

    def on_exception(self, trainer: Trainer, pl_module: LighterModule, exception: BaseException) -> None:
        """Close the file on errors to prevent file handle leaks."""
        self._close_file()

    def teardown(self, trainer: Trainer, pl_module: LighterModule, stage: str) -> None:
        """Guarantee cleanup when stage is PREDICT."""
        if stage == Stage.PREDICT:
            self._close_file()
