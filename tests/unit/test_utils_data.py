from lighter.utils.data import collate_replace_corrupted


def test_collate_replace_corrupted_basic():
    """Test basic functionality of collate_replace_corrupted.

    Tests:
        - Output maintains correct length
        - Non-corrupted values remain unchanged
        - Corrupted values are replaced with valid dataset values
    """
    batch = [1, None, 2, None, 3]
    dataset = [1, 2, 3, 4, 5]
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


def test_collate_replace_corrupted_all_corrupted():
    """Test collate_replace_corrupted handles completely corrupted batch.

    Tests:
        - Batch with all corrupted values is handled correctly
        - Output maintains correct length
        - All values are replaced with valid dataset values
    """
    dataset = [1, 2, 3, 4, 5]
    all_corrupted_batch = [None, None, None]
    collated_all_corrupted = collate_replace_corrupted(all_corrupted_batch, dataset)
    assert len(collated_all_corrupted) == len(all_corrupted_batch)
    assert all(val in dataset for val in collated_all_corrupted)
