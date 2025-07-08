"""
This module provides utility functions for callbacks, including mode conversion and image preprocessing.
"""

import torch
import torchvision
from torch import Tensor


def preprocess_image(image: Tensor) -> Tensor:
    """
    Preprocess image for logging. For multiple 2D images, creates a grid.
    For 3D images, stacks slices vertically. For multiple 3D images, creates a grid
    with each column showing a different 3D image stacked vertically.

    Args:
        image: A 2D or 3D image tensor.

    Returns:
        Tensor: The preprocessed image ready for logging.
    """
    # If 3D (BCDHW), concat the images vertically and horizontally.
    if image.ndim == 5:
        shape = image.shape
        # BCDHW -> BC(D*H)W. Combine slices of a 3D images vertically into a single 2D image.
        image = image.view(shape[0], shape[1], shape[2] * shape[3], shape[4])
        # BCDHW -> 1CDH(B*W). Concat images in the batch horizontally, and unsqueeze to add back the B dim.
        image = torch.cat([*image], dim=-1).unsqueeze(0)
    # If only one image in the batch, select it and return it. Same happens when the images are 3D as they
    # are combined into a single image. `make_grid` is called when a batch of multiple 2D image is provided.
    return image[0] if image.shape[0] == 1 else torchvision.utils.make_grid(image, nrow=8)
