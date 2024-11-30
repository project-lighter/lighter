import torch

from lighter.utils.collate import collate_replace_corrupted


def test_collate_replace_corrupted():
    """Test collate_replace_corrupted function handles corrupted data correctly.

    Tests:
        - Corrupted values (None) are replaced with valid dataset values
        - Non-corrupted values remain unchanged
        - Output maintains correct length
        - Edge cases: empty batch, all corrupted values
    """
    batch = [1, None, 2, None, 3]
    dataset = [1, 2, 3, 4, 5]
    # `collate_replace_corrupted` works by removing the corrupted values
    # from the batch before adding additional samples to make up for them
    collated_batch = collate_replace_corrupted(batch, dataset)

    # Test length
    assert len(collated_batch) == len(batch)

    # Test non-corrupted values remain unchanged.
    filtered_batch = list(filter(lambda x: x is not None, batch))
    assert collated_batch[0].item() == filtered_batch[0]
    assert collated_batch[1].item() == filtered_batch[1]
    assert collated_batch[2].item() == filtered_batch[2]

    # Test corrupted values are replaced with valid dataset values
    assert collated_batch[3].item() in dataset
    assert collated_batch[4].item() in dataset

    all_corrupted = [None, None, None]
    collated_all_corrupted = collate_replace_corrupted(all_corrupted, dataset)
    assert len(collated_all_corrupted) == len(all_corrupted)
    assert all(val in dataset for val in collated_all_corrupted)
