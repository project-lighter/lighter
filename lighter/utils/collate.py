import random

import torch
from torch.utils.data import DataLoader


def collate_fn_replace_corrupted(batch: torch.Tensor, dataset: DataLoader) -> torch.Tensor:
    """Collate function that allows to replace corrupted examples in the batch.
    The dataloader should return `None` when that occurs.
    The `None`s in the batch are replaced with other, randomly-selected, examples.

    Args:
        batch (torch.Tensor): batch from the DataLoader.
        dataset (Dataset): dataset that the DataLoader is passing through. Needs to be fixed
            in place with functools.partial before passing it to DataLoader's 'collate_fn' option
            as 'collate_fn' should only have a single argument - batch. Example:
                ```
                collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=dataset)`
                loader = DataLoader(dataset, ..., collate_fn=collate_fn).
                ```
    Returns:
        torch.Tensor: batch with new examples instead of corrupted ones.
    """
    # Idea from https://stackoverflow.com/a/57882783
    original_batch_len = len(batch)
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    filtered_batch_len = len(batch)
    # Num of corrupted examples
    num_corrupted = original_batch_len - filtered_batch_len
    if num_corrupted > 0:
        # Replace a corrupted example with another randomly selected example
        batch.extend([dataset[random.randint(0, len(dataset))] for _ in range(num_corrupted)])
        # Recursive call to replace the replacements if they are corrupted
        return collate_fn_replace_corrupted(batch, dataset)
    # Finally, when the whole batch is fine, return it
    return torch.utils.data.dataloader.default_collate(batch)
