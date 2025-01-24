"""
This module provides custom collate functions and a DataLoader factory for PyTorch.

Key components:
- collate_replace_corrupted: A function to handle corrupted examples in batches by replacing them with valid ones
- DataLoader: A factory class that creates PyTorch DataLoaders with mode-specific configurations
"""

from typing import Any, Callable

import inspect
import random

import torch
from loguru import logger
from torch.utils.data.dataloader import default_collate

from lighter.utils.types.enums import Mode


def collate_replace_corrupted(
    batch: Any, dataset: torch.utils.data.Dataset, default_collate_fn: Callable | None = None
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
    """A factory class for delayed PyTorch DataLoader instantiation."""

    def __init__(self, cls=torch.utils.data.DataLoader, *args, **kwargs):
        """
        Initialize the DataLoader factory with arguments for delayed instantiation.

        Args:
            cls: The DataLoader class to instantiate (default: torch.utils.data.DataLoader)
            *args: Positional arguments for the DataLoader.
            **kwargs: Keyword arguments for the DataLoader.
        """
        signature = inspect.signature(cls)
        # Assign args and kwargs to the parameters. This way, we essentially turn the args into kwargs.
        try:
            kwargs = signature.bind_partial(*args, **kwargs).arguments
        # Throws a TypeError if the arguments do not match the signature
        except TypeError as e:
            raise TypeError(f"{cls} {e}") from e
        # Use a separate attribute to store parameters to avoid conflicts
        object.__setattr__(self, "_kwargs", kwargs)
        object.__setattr__(self, "cls", cls)

    def __call__(self, mode) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader with mode-specific configurations.

        Args:
            mode: The mode to operate in (train, val, test, predict)

        Returns:
            A configured PyTorch DataLoader
        """
        # Handle shuffling when not specified if the DataLoader class has a `shuffle` parameter
        if "shuffle" not in self._kwargs and "shuffle" in inspect.signature(self.cls).parameters:
            shuffle = mode == Mode.TRAIN
            self._kwargs["shuffle"] = shuffle
            logger.warning(f"Setting 'shuffle={shuffle}' for DataLoader in '{mode}' mode since it was not specified.")

        # Disable `drop_last` if set to True in val/test/predict modes
        if mode != Mode.TRAIN and self._kwargs.get("drop_last", False):
            self._kwargs["drop_last"] = False
            logger.warning(f"Setting 'drop_last=False' for DataLoader in '{mode}' mode to avoid dropping data.")

        return self.cls(**self._kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        # Protect internal attributes from being overwritten
        if "_kwargs" in self.__dict__ and name in self._kwargs:
            self._kwargs[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        # Expose attributes from the internal kwargs. For example, self.batch_size instead of self._kwargs["batch_size"].
        if "_kwargs" in self.__dict__ and name in self._kwargs:
            return self._kwargs[name]
        else:
            return object.__getattribute__(self, name)
