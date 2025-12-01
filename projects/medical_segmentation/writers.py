"""Custom writer functions for medical imaging outputs."""

from pathlib import Path

import torch


def write_nrrd(path: Path, tensor: torch.Tensor, *, suffix: str = ".seg.nrrd") -> None:
    """Write a 3D segmentation tensor to NRRD format.

    Uses nibabel for NIfTI/NRRD I/O which is standard in medical imaging.
    NRRD (Nearly Raw Raster Data) is commonly used for medical segmentation masks.

    Args:
        path: Output path (suffix will be replaced).
        tensor: 3D tensor (D, H, W) containing segmentation labels.
        suffix: File suffix (default: .seg.nrrd for segmentation convention).
    """
    import nibabel as nib
    import numpy as np

    path = path.with_suffix(suffix)

    # Convert tensor to numpy
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy()
    else:
        data = np.asarray(tensor)

    # Remove batch/channel dims if present: (B, C, D, H, W) -> (D, H, W)
    while data.ndim > 3:
        data = data[0]

    # Create NIfTI image (NRRD is handled by nibabel via nrrd backend)
    # Use identity affine - in practice, you'd want to preserve the original affine
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(data.astype(np.int16), affine)

    # Save - nibabel determines format from suffix
    nib.save(nifti_img, str(path))
