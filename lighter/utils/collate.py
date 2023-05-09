from typing import Any, Callable, List

import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate_str_fn, default_collate_fn_map
from torch.utils.data.dataloader import default_collate

# Collate support for None. Just as a string, None is not collated. Allows elements of the batch to be None.
default_collate_fn_map.update({type(None): collate_str_fn})


def collate_replace_corrupted(batch: Any, dataset: DataLoader, default_collate_fn: Callable = None) -> Any:
    """Collate function that allows to replace corrupted examples in the batch.
    The dataloader should return `None` when that occurs.
    The `None`s in the batch are replaced with other, randomly-selected, examples.

    Args:
        batch (Any): batch from the DataLoader.
        dataset (Dataset): dataset that the DataLoader is passing through. Needs to be fixed
            in place with functools.partial before passing it to DataLoader's 'collate_fn' option
            as 'collate_fn' should only have a single argument - batch. Example:
                ```
                collate_fn = functools.partial(collate_replace_corrupted, dataset=dataset)`
                loader = DataLoader(dataset, ..., collate_fn=collate_fn).
                ```
        default_collate_fn (Callable): the collate function to call once the batch has no corrupted examples.
            If `None`, `torch.utils.data.dataloader.default_collate` is called. Defaults to None.
    Returns:
        Any: batch with new examples instead of corrupted ones.
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
