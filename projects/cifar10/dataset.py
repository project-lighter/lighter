"""CIFAR-10 with a validation set held out from the training images, so the test images are used only for testing."""

import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10


def cifar10_split(
    root: str,
    split: str,
    transform=None,
    target_transform=None,
    download: bool = True,
    validation_fraction: float = 0.1,
    split_seed: int = 42,
) -> Dataset:
    """`train` and `val` are disjoint parts of the 50,000 training images, chosen by `split_seed`; `test` is the
    10,000 test images."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")
    dataset = CIFAR10(root, train=split != "test", transform=transform, target_transform=target_transform, download=download)
    if split == "test":
        return dataset
    count = int(len(dataset) * validation_fraction)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(split_seed)).tolist()
    return Subset(dataset, indices[count:] if split == "train" else indices[:count])
