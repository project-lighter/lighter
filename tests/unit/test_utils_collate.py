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
    collated_batch = collate_replace_corrupted(batch, dataset)
    # Test length
    assert len(collated_batch) == len(batch)
    
    # Test non-corrupted values remain unchanged
    assert collated_batch[0] == 1
    assert collated_batch[2] == 2
    assert collated_batch[4] == 3
    
    # Test corrupted values are replaced with valid dataset values
    assert collated_batch[1] in dataset
    assert collated_batch[3] in dataset
    
    # Test edge cases
    empty_batch = []
    assert len(collate_replace_corrupted(empty_batch, dataset)) == 0
    
    all_corrupted = [None, None, None]
    collated_all_corrupted = collate_replace_corrupted(all_corrupted, dataset)
    assert len(collated_all_corrupted) == len(all_corrupted)
    assert all(val in dataset for val in collated_all_corrupted)
