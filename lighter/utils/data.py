from typing import Any, Callable

import random

import torch
from torch.utils.data.dataloader import default_collate


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
