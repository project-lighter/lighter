"""Datasets for self-supervised learning using lightly library.

Lightly provides LightlyDataset and transforms optimized for SSL methods.
"""

from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform
from torchvision import datasets

__all__ = ["LightlyDataset", "SimCLRTransform", "get_cifar10_ssl_dataset", "get_stl10_ssl_dataset"]


def get_cifar10_ssl_dataset(
    root: str = "./.datasets/",
    train: bool = True,
    input_size: int = 32,
    download: bool = True,
) -> LightlyDataset:
    """Create CIFAR10 dataset with SSL augmentations.

    Args:
        root: Root directory for dataset.
        train: Use training split.
        input_size: Input image size for transforms.
        download: Download dataset if not present.

    Returns:
        LightlyDataset with SimCLR transforms.
    """
    # Download CIFAR10 to get the data files
    base_dataset = datasets.CIFAR10(root=root, train=train, download=download)

    # Create transform for SSL
    transform = SimCLRTransform(input_size=input_size)

    # LightlyDataset wraps the base dataset
    return LightlyDataset.from_torch_dataset(base_dataset, transform=transform)


def get_stl10_ssl_dataset(
    root: str = "./.datasets/",
    split: str = "unlabeled",
    input_size: int = 96,
    download: bool = True,
) -> LightlyDataset:
    """Create STL10 dataset with SSL augmentations.

    STL10 has an 'unlabeled' split (100k images) ideal for SSL pretraining.

    Args:
        root: Root directory for dataset.
        split: Dataset split ('train', 'test', 'unlabeled').
        input_size: Input image size for transforms.
        download: Download dataset if not present.

    Returns:
        LightlyDataset with SimCLR transforms.
    """
    base_dataset = datasets.STL10(root=root, split=split, download=download)
    transform = SimCLRTransform(input_size=input_size)
    return LightlyDataset.from_torch_dataset(base_dataset, transform=transform)
