"""Medical imaging datasets using MONAI.

MONAI provides ready-to-use medical imaging datasets including:
- DecathlonDataset: Medical Segmentation Decathlon (10 tasks)
- MedNISTDataset: Medical version of MNIST
- TciaDataset: The Cancer Imaging Archive datasets
"""

from monai.apps import DecathlonDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRanged,
    Spacingd,
)

__all__ = [
    "DecathlonDataset",
    "get_spleen_train_transforms",
    "get_spleen_val_transforms",
]


def get_spleen_train_transforms(
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    num_samples: int = 4,
) -> Compose:
    """Get training transforms for spleen segmentation.

    Args:
        spatial_size: Size of random crops.
        num_samples: Number of crops per volume.

    Returns:
        MONAI Compose transform.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=num_samples,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_spleen_val_transforms() -> Compose:
    """Get validation/test transforms for spleen segmentation.

    Returns:
        MONAI Compose transform.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
