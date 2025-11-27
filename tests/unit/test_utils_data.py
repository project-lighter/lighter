import pytest

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


def test_collate_replace_corrupted_max_retries():
    """Test that max_retries prevents infinite loops.

    Tests:
        - Function raises RuntimeError when max_retries is exceeded
        - Error message provides helpful information about corruption rate
        - Function works correctly with custom max_retries parameter
    """

    # Create a dataset that always returns None (fully corrupted)
    class CorruptedDataset:
        def __getitem__(self, idx):
            return None

        def __len__(self):
            return 100

    dataset = CorruptedDataset()
    batch = [None, None, None]

    # Test with low max_retries to trigger the error quickly
    with pytest.raises(RuntimeError) as exc_info:
        collate_replace_corrupted(batch, dataset, max_retries=5)

    # Verify the error message contains helpful information
    error_msg = str(exc_info.value)
    assert "maximum retry limit (5)" in error_msg
    assert "high corruption rate" in error_msg
    assert "increasing max_retries" in error_msg


def test_collate_replace_corrupted_custom_max_retries():
    """Test that custom max_retries parameter works correctly.

    Tests:
        - Function respects custom max_retries value
        - Function succeeds when valid samples are eventually found
    """

    # Create a dataset that returns None for first few accesses, then valid data
    class PartiallyCorruptedDataset:
        def __init__(self):
            self.access_count = 0

        def __getitem__(self, idx):
            self.access_count += 1
            # Return None for first 10 accesses, then valid data
            return None if self.access_count <= 10 else 42

        def __len__(self):
            return 100

    dataset = PartiallyCorruptedDataset()
    batch = [None, None]

    # This should eventually succeed with enough retries
    result = collate_replace_corrupted(batch, dataset, max_retries=20)
    assert len(result) == 2
    # At least some values should be 42 (the valid replacement)
    assert any(val.item() == 42 for val in result)
