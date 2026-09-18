"""Explicit CIFAR10 research splits; the official test set stays held out."""

from collections.abc import Callable
from typing import cast

import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10


def cifar10_split(
    root: str,
    split: str,
    transform: Callable | None = None,
    target_transform: Callable | None = None,
    download: bool = True,
    validation_fraction: float = 0.1,
    split_seed: int = 42,
) -> Dataset:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    dataset = CIFAR10(
        root=root, train=split != "test", transform=transform, target_transform=target_transform, download=download
    )
    if split == "test":
        return cast(Dataset, dataset)
    count = int(len(dataset) * validation_fraction)
    if not 0 < count < len(dataset):
        raise ValueError("validation_fraction must give nonempty training and validation populations")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(split_seed)).tolist()
    return Subset(dataset, indices[count:] if split == "train" else indices[:count])
