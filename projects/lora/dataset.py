"""Datasets for efficient fine-tuning examples."""

from collections.abc import Callable

import torch
from torch.utils.data import Dataset
from torchvision import datasets


class CIFAR100Dataset(Dataset):
    """CIFAR100 dataset wrapper for fine-tuning.

    CIFAR100 is useful for demonstrating fine-tuning because:
    - 100 classes provide a challenging classification task
    - Images are small enough for quick experimentation
    - Pretrained models on ImageNet transfer well

    Args:
        root: Root directory for dataset.
        split: "train" or "val" (disjoint parts of the training images, chosen by `split_seed`), or "test".
        transform: Transform to apply to images.
        download: Whether to download if not present.
        validation_fraction: Fraction of the training images held out for validation.
        split_seed: Seed that chooses the validation images; keep it fixed when comparing runs.
    """

    def __init__(
        self,
        root: str = "./.datasets/",
        split: str = "train",
        transform: Callable | None = None,
        download: bool = True,
        validation_fraction: float = 0.1,
        split_seed: int = 42,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")
        self.dataset = datasets.CIFAR100(root=root, train=split != "test", download=download)
        self.transform = transform
        self.indices = list(range(len(self.dataset)))
        if split != "test":
            count = int(len(self.dataset) * validation_fraction)
            indices = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(split_seed)).tolist()
            self.indices = indices[count:] if split == "train" else indices[:count]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[self.indices[idx]]

        if self.transform:
            image = self.transform(image)

        return image, label


class FewShotDataset(Dataset):
    """Few-shot learning dataset for fine-tuning evaluation.

    Creates a dataset with limited samples per class, which is
    the typical scenario where LoRA shines.

    Args:
        base_dataset: The full dataset to sample from.
        samples_per_class: Number of samples per class (k-shot).
        num_classes: Number of classes to include (n-way).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        samples_per_class: int = 5,
        num_classes: int = 10,
        seed: int = 42,
    ) -> None:
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes

        # Group samples by class
        class_indices: dict[int, list[int]] = {}
        for idx in range(len(base_dataset)):
            _, label = base_dataset[idx]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # Sample k examples from n classes
        generator = torch.Generator().manual_seed(seed)
        selected_classes = list(class_indices.keys())[:num_classes]

        self.indices = []
        self.label_map = {old: new for new, old in enumerate(selected_classes)}

        for cls in selected_classes:
            cls_indices = class_indices[cls]
            perm = torch.randperm(len(cls_indices), generator=generator)
            selected = perm[:samples_per_class].tolist()
            self.indices.extend([cls_indices[i] for i in selected])

        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        image, old_label = self.base_dataset[real_idx]
        new_label = self.label_map[old_label]
        return image, new_label
