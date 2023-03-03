import torch

from lighter.utils.collate import collate_fn_replace_corrupted


def test_collate_fn_replace_corrupted():
    # Test when there are no corrupted examples in the batch
    dataset1 = [torch.randn(3, 224, 224) for _ in range(10)]
    batch1 = [dataset1[0], dataset1[1], dataset1[2]]
    new_batch1 = collate_fn_replace_corrupted(batch1, dataset=dataset1)
    assert len(new_batch1) == 3
    assert torch.is_tensor(new_batch1)

    # Test when all examples in the batch are corrupted
    dataset2 = [torch.randn(3, 224, 224) for _ in range(10)]
    batch2 = [None, None, None]
    new_batch2 = collate_fn_replace_corrupted(batch2, dataset=dataset2)
    assert len(new_batch2) == 3
    assert torch.is_tensor(new_batch2)

    # Test when there is one corrupted examples in the batch
    dataset3 = [torch.randn(3, 224, 224) for _ in range(10)]
    batch3 = [dataset3[0], None, dataset3[2]]
    new_batch3 = collate_fn_replace_corrupted(batch3, dataset=dataset3)
    assert len(new_batch3) == 3
    assert torch.is_tensor(new_batch3)
