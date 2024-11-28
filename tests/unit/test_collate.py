import pytest
from lighter.utils.collate import collate_replace_corrupted

def test_collate_replace_corrupted():
    batch = [1, None, 2, None, 3]
    dataset = [1, 2, 3, 4, 5]
    collated_batch = collate_replace_corrupted(batch, dataset)
    assert len(collated_batch) == 5
