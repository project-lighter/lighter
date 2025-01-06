"""
This module provides the FileWriter class, which writes predictions to files in various formats.
"""

from typing import Callable

from functools import partial

import torch
import torchvision
from monai.data import MetaTensor, metatensor_to_itk_image
from monai.transforms import DivisiblePad
from torch import Tensor

from lighter.callbacks.utils import preprocess_image
from lighter.callbacks.writer.base import BaseWriter
from lighter.utils.dynamic_imports import OPTIONAL_IMPORTS


class FileWriter(BaseWriter):
    """
    Writer for saving predictions to files in various formats including tensors, images, videos, and ITK images.
    Custom writer functions can be provided to extend supported formats.
    Args:
        path: Directory path where output files will be saved.
        writer: Either a string specifying a built-in writer or a custom writer function.
            Built-in writers:
                - "tensor": Saves raw tensor data (.pt)
                - "image": Saves as image file (.png)
                - "video": Saves as video file
                - "itk_nrrd": Saves as ITK NRRD file (.nrrd)
                - "itk_seg_nrrd": Saves as ITK segmentation NRRD file (.seg.nrrd)
                - "itk_nifti": Saves as ITK NIfTI file (.nii.gz)
            Custom writers must:
                - Accept (path, tensor) arguments
                - Handle single tensor input (no batch dimension)
                - Save output to the specified path
    """

    @property
    def writers(self) -> dict[str, Callable]:
        return {
            "tensor": write_tensor,
            "image": write_image,
            "video": write_video,
            "itk_nrrd": partial(write_itk_image, suffix=".nrrd"),
            "itk_seg_nrrd": partial(write_itk_image, suffix=".seg.nrrd"),
            "itk_nifti": partial(write_itk_image, suffix=".nii.gz"),
        }

    def write(self, tensor: Tensor, identifier: int | str) -> None:
        """
        Writes the tensor to a file using the specified writer.

        Args:
            tensor: The tensor to write.
            identifier: Identifier for naming the file.
        """
        if not self.path.is_dir():
            raise RuntimeError(f"FileWriter expects a directory path, got {self.path}")

        # Determine the path for the file based on prediction count. The suffix must be added by the writer function.
        path = self.path / str(identifier)
        # Write the tensor to the file.
        self.writer(path, tensor)


def write_tensor(path, tensor):
    """
    Writes a tensor to a file in .pt format.

    Args:
        path: The path to save the tensor.
        tensor: The tensor to save.
    """
    torch.save(tensor, path.with_suffix(".pt"))  # nosec B614


def write_image(path, tensor):
    """
    Writes a tensor as an image file in .png format.

    Args:
        path: The path to save the image.
        tensor: The tensor representing the image.
    """
    path = path.with_suffix(".png")
    tensor = preprocess_image(tensor)
    torchvision.io.write_png(tensor, str(path))


def write_video(path, tensor):
    """
    Writes a tensor as a video file in .mp4 format.

    Args:
        path: The path to save the video.
        tensor: The tensor representing the video.
    """
    path = path.with_suffix(".mp4")
    # Video tensor must be divisible by 2. Pad the height and width.
    tensor = DivisiblePad(k=(0, 2, 2), mode="minimum")(tensor)
    # Video tensor must be THWC. Permute CTHW -> THWC.
    tensor = tensor.permute(1, 2, 3, 0)
    # Video tensor must have 3 channels (RGB). Repeat the channel dim to convert grayscale to RGB.
    if tensor.shape[-1] == 1:
        tensor = tensor.repeat(1, 1, 1, 3)
    # Video tensor must be in the range [0, 1]. Scale to [0, 255].
    tensor = (tensor * 255).to(torch.uint8)
    torchvision.io.write_video(str(path), tensor, fps=24)


def write_itk_image(path: str, tensor: MetaTensor, suffix) -> None:
    """
    Writes a tensor as an ITK image file.

    Args:
        path: The path to save the ITK image.
        tensor: The tensor representing the image. Must be in MONAI MetaTensor format.
        suffix: The file suffix indicating the format.
    """
    path = path.with_suffix(suffix)
    if not isinstance(tensor, MetaTensor):
        raise TypeError("Tensor must be in MONAI MetaTensor format.")
    itk_image = metatensor_to_itk_image(tensor, channel_dim=0, dtype=tensor.dtype)
    OPTIONAL_IMPORTS["itk"].imwrite(itk_image, str(path), True)
