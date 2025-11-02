import random
from typing import Any, Callable

import torch
from loguru import logger
from torch.utils.data.dataloader import default_collate


def collate_replace_corrupted(
    batch: list[Any], dataset: torch.utils.data.Dataset, default_collate_fn: Callable | None = None
) -> Any:
    """
    Collate function that handles corrupted examples in a batch by replacing them with valid ones.

    This function is designed to prevent training interruptions due to data corruption.
    It logs a warning to alert the user about the number of corrupted samples found.

    Args:
        batch: The batch of data from the DataLoader.
        dataset: The dataset being used, which should return `None` for corrupted examples.
        default_collate_fn: The default collate function to use once the batch is clean.

    Returns:
        A batch with corrupted examples replaced by valid ones.
    """
    # Use `torch.utils.data.dataloader.default_collate` if no other default collate function is specified.
    default_collate_fn = default_collate_fn if default_collate_fn is not None else default_collate

    num_corrupted = 0
    while True:
        # Filter out corrupted samples (None).
        original_len = len(batch)
        batch = [sample for sample in batch if sample is not None]
        current_len = len(batch)

        # Calculate the number of corrupted samples in this iteration.
        newly_corrupted = original_len - current_len
        if newly_corrupted == 0:
            # No more corrupted samples, break the loop.
            break

        num_corrupted += newly_corrupted

        # Replace corrupted samples with new random samples from the dataset.
        replacements = [dataset[random.randint(0, len(dataset) - 1)] for _ in range(newly_corrupted)]
        batch.extend(replacements)

    # Log a warning if any corrupted samples were found and replaced.
    if num_corrupted > 0:
        logger.warning(f"Found and replaced {num_corrupted} corrupted samples in a batch.")

    # Apply the default collate function to the clean batch.
    return default_collate_fn(batch)
