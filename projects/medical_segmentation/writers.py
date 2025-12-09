"""Custom writer functions for medical imaging outputs."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch


def write_nrrd(
    path: Path,
    tensor: torch.Tensor,
    *,
    suffix: str = ".seg.nrrd",
    affine: np.ndarray | None = None,
    spatial_shape: Sequence[int] | None = None,
) -> None:
    """Write a 3D segmentation tensor to NRRD format.

    Uses MONAI's ITKWriter for robust I/O with proper orientation handling.
    NRRD (Nearly Raw Raster Data) is commonly used for medical segmentation masks.

    Args:
        path: Output path (suffix will be replaced).
        tensor: 3D tensor (D, H, W) or (C, D, H, W) containing segmentation labels.
        suffix: File suffix (default: .seg.nrrd for segmentation convention).
        affine: 4x4 affine matrix for spatial orientation. If None, uses identity.
        spatial_shape: Original spatial shape for resampling. If None, uses tensor shape.
    """
    from monai.data.image_writer import ITKWriter

    path = path.with_suffix(suffix)

    # Convert tensor to numpy
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy()
    else:
        data = np.asarray(tensor)

    # Remove batch dim if present: (B, C, D, H, W) -> (C, D, H, W) or (D, H, W)
    if data.ndim == 5:
        if data.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {data.shape[0]}. Use batch_size=1 for prediction.")
        data = data[0]

    # Validate final shape is 3D or 4D
    if data.ndim not in (3, 4):
        raise ValueError(f"Expected 3D (D,H,W) or 4D (C,D,H,W) data after removing batch, got shape {data.shape}")

    # Determine channel_dim: if 4D assume first dim is channel, else no channel
    channel_dim = 0 if data.ndim == 4 else None

    # Use identity affine if not provided
    if affine is None:
        affine = np.eye(4)

    writer = ITKWriter(output_dtype=np.int16)
    writer.set_data_array(data, channel_dim=channel_dim)
    writer.set_metadata({"affine": affine, "spatial_shape": spatial_shape or data.shape[-3:]})
    writer.write(str(path))
