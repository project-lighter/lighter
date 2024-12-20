"""
This module provides custom collate functions and a DataLoader factory for PyTorch.

Key components:
- collate_replace_corrupted: A function to handle corrupted examples in batches by replacing them with valid ones
- DataLoader: A factory class that creates PyTorch DataLoaders with mode-specific configurations
"""

from typing import Any, Callable

import random
from functools import partial

import torch
from torch.utils.data import Sampler
from torch.utils.data._utils.collate import collate_str_fn, default_collate_fn_map
from torch.utils.data.dataloader import default_collate

from lighter.engine.schema import ModeSchema

# Collate support for None. Just as a string, None is not collated. Allows elements of the batch to be None.
default_collate_fn_map.update({type(None): collate_str_fn})


def collate_replace_corrupted(
    batch: Any, dataset: torch.utils.data.DataLoader, default_collate_fn: Callable | None = None
) -> Any:
    """
    Collate function to handle corrupted examples in a batch by replacing them with valid ones.

    Args:
        batch: The batch of data from the DataLoader.
        dataset: The dataset being used, which should return `None` for corrupted examples.
        default_collate_fn: The default collate function to use once the batch is clean.

    Returns:
        A batch with corrupted examples replaced by valid ones.
    """
    # Use `torch.utils.data.dataloader.default_collate` if no other default collate function is specified.
    default_collate_fn = default_collate_fn if default_collate_fn is not None else default_collate
    # Idea from https://stackoverflow.com/a/57882783
    original_batch_len = len(batch)
    # Filter out all the Nones (corrupted examples).
    batch = list(filter(lambda x: x is not None, batch))
    filtered_batch_len = len(batch)
    # Num of corrupted examples.
    num_corrupted = original_batch_len - filtered_batch_len
    if num_corrupted > 0:
        # Replace a corrupted example with another randomly selected example.
        batch.extend([dataset[random.randint(0, len(dataset) - 1)] for _ in range(num_corrupted)])
        # Recursive call to replace the replacements if they are corrupted.
        return collate_replace_corrupted(batch, dataset)
    # Finally, when the whole batch is fine, apply the default collate function.
    return default_collate_fn(batch)


class DataLoader:
    """A DataLoader factory that creates PyTorch DataLoaders with mode-specific configurations.


    DataLoader configurations often use the same settings across different modes (train, val, test, predict).
    This class allows specifying shared settings once, while still supporting mode-specific overrides when needed.

    Each DataLoader parameter can be specified in two ways:
    1. As a single value that applies across all modes:
        ```python
        batch_size=32  # Uses batch size 32 for all modes
        ```
    2. As a dictionary mapping specific modes to different values:
        ```python
        batch_size={"train": 32, "val": 16}  # Uses batch size 32 for training and 16 for validation
        ```
    Note: Arguments match the PyTorch DataLoader except for `dataset` and `shuffle`, which are configured internally.

    Args:
        batch_size: Number of samples per batch. Either an int for all modes or dict mapping modes to batch sizes.
        num_workers: Number of subprocesses for data loading. Default: 0 (main process only).
        sampler: Defines strategy to draw samples from dataset. Default: None.
        batch_sampler: Alternative to sampler that yields batches directly. Default: None.
        collate_fn: Merges list of samples into batch. Default: None uses default_collate.
        pin_memory: Copy Tensors into CUDA pinned memory. Default: False.
        drop_last: Drop last incomplete batch in training. Default: False.
        timeout: Timeout for collecting batch. Default: 0 (no timeout).
        worker_init_fn: Function called for each worker process. Default: None.
        multiprocessing_context: Multiprocessing implementation to use. Default: None.
        generator: Generator for random number sampling. Default: None.
        prefetch_factor: Number of batches loaded per worker. Default: None.
        persistent_workers: Reuse workers between epochs. Default: False.
        pin_memory_device: Device to pin memory to. Default: None.

    Examples:
        ```python
        # All modes use the same settings
        loader = DataLoader(
            batch_size=32,
            num_workers=4,
            pin_memory=True
        )

        # Different settings per mode
        loader = DataLoader(
            batch_size={"train": 32, "val": 16, "test": 16, "predict": 1},
            num_workers={"train": 4, "val": 2, "test": 2, "predict": 1},
            pin_memory={"train": True, "val": False, "test": False, "predict": False}
        )

        # Mix of shared and mode-specific settings
        loader = DataLoader(
            batch_size={"train": 32, "val": 16},  # Different batch sizes for train/val
            num_workers=4,                         # Same number of workers for all modes
            pin_memory=True                        # Same pin_memory setting for all modes
        )
        ```
    """

    def __init__(
        self,
        batch_size: int | dict[str, int],
        num_workers: int | dict[str, int] = 0,
        sampler: Sampler | dict[str, Sampler] | None = None,
        batch_sampler: Sampler | dict[str, Sampler] | None = None,
        collate_fn: callable | dict[str, callable] | None = None,
        pin_memory: bool | dict[str, bool] = False,
        drop_last: bool | dict[str, bool] = False,
        timeout: float | dict[str, float] = 0,
        worker_init_fn: callable | dict[str, callable] | None = None,
        multiprocessing_context: str | dict[str, str] | None = None,
        generator: torch.Generator | dict[str, torch.Generator] | None = None,
        prefetch_factor: int | dict[str, int] | None = None,
        persistent_workers: bool | dict[str, bool] = False,
        pin_memory_device: str | dict[str, str] | None = None,
    ):
        # Store all dataloader settings, supporting both global and mode-specific values
        self.settings = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "sampler": sampler,
            "batch_sampler": batch_sampler,
            "collate_fn": collate_fn,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
            "timeout": timeout,
            "worker_init_fn": worker_init_fn,
            "multiprocessing_context": multiprocessing_context,
            "generator": generator,
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
            "pin_memory_device": pin_memory_device,
        }

        # Validate that any mode-specific settings only use valid mode names
        for key, value in self.settings.items():
            if isinstance(value, dict):
                invalid_modes = set(value.keys()) - ModeSchema.model_fields
                if invalid_modes:
                    raise ValueError(
                        f"Invalid modes {invalid_modes} found in {key}. Valid modes are: {ModeSchema.model_fields}"
                    )

    def __call__(self, dataset, mode: str, inferer=None) -> torch.utils.data.DataLoader:
        """Creates a PyTorch DataLoader configured for the specified mode.

        Creates a DataLoader with the appropriate settings for the given mode, handling both
        global settings and mode-specific overrides.

        When an inferer is provided, verify that batch_size=1, since this is required by Lighter's
        inference system.

        Args:
            dataset: The dataset to load data from
            mode: The mode to create the loader for ('train', 'val', 'test', or 'predict')
            inferer: Optional inferer for validation/test/predict modes. If provided, validates
                    that batch_size=1.

        Returns:
            A configured PyTorch DataLoader instance

        Raises:
            ValueError: If mode-specific settings are missing the requested mode or if
                      batch_size != 1 with an inferer in val/test/predict modes
        """
        # Extract mode-specific or global settings
        settings = {}
        for key, value in self.settings.items():
            if isinstance(value, dict):
                if mode not in value:
                    raise ValueError(f"DataLoader '{key}' argument is defined as a dict, but key '{mode}' is missing.")
                settings[key] = value[mode]
            else:
                settings[key] = value

        # Raise error if batch_size != 1 when using inferer for val/test/predict
        if inferer is not None and mode in ["val", "test", "predict"] and settings["batch_size"] != 1:
            raise ValueError(f"Batch size must be 1 for '{mode}' mode when an inferer is provided.")

        # Shuffle=True only for training. If a sampler/batch_sampler is provided, shuffle should be handled there.
        if settings["sampler"] is None and settings["batch_sampler"] is None:
            settings["shuffle"] = mode == "train"

        # Wrap collate_fn with corrupted sample handling
        settings["collate_fn"] = partial(collate_replace_corrupted, dataset=dataset, default_collate_fn=settings["collate_fn"])

        return torch.utils.data.DataLoader(dataset=dataset, **settings)
