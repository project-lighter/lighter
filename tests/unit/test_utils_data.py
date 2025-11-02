import random
from unittest.mock import patch

import pytest
import torch
from loguru import logger

from lighter.utils.data import collate_replace_corrupted


class MockDataset(torch.utils.data.Dataset):
    """
    A mock dataset that can return corrupted (None) samples.
    """

    def __init__(self, data, corrupted_indices=None):
        self.data = data
        self.corrupted_indices = set(corrupted_indices) if corrupted_indices else set()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx in self.corrupted_indices:
            return None
        return self.data[idx]


@pytest.fixture
def sample_dataset():
    """
    Fixture providing a sample dataset for testing.
    """
    return MockDataset(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def test_collate_replace_corrupted_no_corruption(sample_dataset):
    """
    Test collate_replace_corrupted with no corrupted samples.
    """
    batch = [1, 2, 3]
    collated_batch = collate_replace_corrupted(batch, sample_dataset)
    assert torch.equal(collated_batch, torch.tensor([1, 2, 3]))


def test_collate_replace_corrupted_some_corruption(sample_dataset):
    """
    Test collate_replace_corrupted with some corrupted samples.
    """
    batch = [1, None, 3, None]
    with patch.object(logger, "warning") as mock_warning:
        collated_batch = collate_replace_corrupted(batch, sample_dataset)
        mock_warning.assert_called_once_with("Found and replaced 2 corrupted samples in a batch.")

    assert len(collated_batch) == 4
    assert collated_batch[0].item() == 1
    assert collated_batch[1].item() in sample_dataset.data
    assert collated_batch[2].item() == 3
    assert collated_batch[3].item() in sample_dataset.data


def test_collate_replace_corrupted_all_corruption(sample_dataset):
    """
    Test collate_replace_corrupted with all corrupted samples.
    """
    batch = [None, None, None]
    with patch.object(logger, "warning") as mock_warning:
        collated_batch = collate_replace_corrupted(batch, sample_dataset)
        mock_warning.assert_called_once_with("Found and replaced 3 corrupted samples in a batch.")

    assert len(collated_batch) == 3
    assert all(val.item() in sample_dataset.data for val in collated_batch)


def test_collate_replace_corrupted_iterative_replacement():
    """
    Test collate_replace_corrupted with iterative replacement (replacements are also corrupted).
    """
    # Dataset where index 0 is corrupted
    dataset = MockDataset(data=[10, 11, 12], corrupted_indices={0})
    # Batch with one corrupted sample, and if replaced by index 0, it will still be corrupted
    batch = [1, None, 3]

    # Mock random.randint to always return 0, so it tries to replace with a corrupted sample
    with patch.object(random, "randint", return_value=0):
        with patch.object(logger, "warning") as mock_warning:
            collated_batch = collate_replace_corrupted(batch, dataset)
            # It should try to replace the first None, get a None, then try again and get a valid one.
            mock_warning.assert_called_once_with("Found and replaced 1 corrupted samples in a batch.")

    assert len(collated_batch) == 3
    assert collated_batch[0].item() == 1
    assert collated_batch[1].item() in dataset.data  # Should eventually get a non-corrupted sample
    assert collated_batch[2].item() == 3
