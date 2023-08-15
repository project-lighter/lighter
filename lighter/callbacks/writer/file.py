from typing import Callable, Dict, Optional, Union

from pathlib import Path

import torch
import torchvision
from monai.transforms import DivisiblePad
from monai.utils.module import optional_import

from lighter.callbacks.utils import preprocess_image
from lighter.callbacks.writer.base import LighterBaseWriter
from lighter.utils.dynamic_imports import OPTIONAL_IMPORTS


class LighterFileWriter(LighterBaseWriter):
    """
    Writer for writing predictions to files. Supports multiple formats, and
    additional custom formats can be added either through `additional_writers`
    argument at initialization, or by calling `add_writer` method after initialization.

    Args:
        directory (Union[str, Path]): The directory where the files should be written.
        format (str): The format in which the files should be saved.
        additional_writers (Optional[Dict[str, Callable]]): Additional custom writer functions.
    """

    def __init__(
        self, directory: Union[str, Path], format: str, additional_writers: Optional[Dict[str, Callable]] = None
    ) -> None:
        # Predefined writers for different formats.
        self._writers = {
            "tensor": write_tensor,
            "image": write_image,
            "video": write_video,
            "sitk_nrrd": write_sitk_nrrd,
            "sitk_seg_nrrd": write_seg_nrrd,
            "sitk_nifti": write_sitk_nifti,
        }
        # Initialize the base class.
        super().__init__(directory, format, additional_writers)

    def write(self, tensor: torch.Tensor, id: Union[int, str], multi_pred_id: Optional[Union[int, str]], format: str) -> None:
        """
        Write the tensor to the specified path in the given format.

        If there are multiple predictions, a directory named `id` is created, and each file is named
        after `multi_pred_id`. If there's a single prediction, the file is named after `id`.

        Args:
            tensor (Tensor): The tensor to be written.
            id (Union[int, str]): The primary identifier for naming.
            multi_pred_id (Optional[Union[int, str]]): The secondary identifier, used if there are multiple predictions.
            format (str): Format in which tensor should be written.
        """
        # Determine the path for the file based on prediction count. The suffix must be added by the writer function.
        path = self.directory / str(id) if multi_pred_id is None else self.directory / str(id) / str(multi_pred_id)
        path.parent.mkdir(exist_ok=True, parents=True)
        # Fetch the appropriate writer function for the format.
        writer = self.get_writer(format)
        # Write the tensor to the file.
        writer(path, tensor)


def write_tensor(path, tensor):
    torch.save(tensor, path.with_suffix(".pt"))


def write_image(path, tensor):
    path = path.with_suffix(".png")
    tensor = preprocess_image(tensor, add_batch_dim=True)
    torchvision.io.write_png(tensor, path)


def write_video(path, tensor):
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


def _write_sitk_image(path: str, tensor: torch.Tensor, suffix) -> None:
    path = path.with_suffix(suffix)

    if "sitk" not in OPTIONAL_IMPORTS:
        OPTIONAL_IMPORTS["sitk"], sitk_available = optional_import("SimpleITK")
        if not sitk_available:
            raise ImportError("SimpleITK is not available. Install it with `pip install SimpleITK`.")

    # Remove the channel dimension if it's equal to 1.
    tensor = tensor.squeeze(0) if (tensor.dim() == 4 and tensor.shape[0] == 1) else tensor
    sitk_image = OPTIONAL_IMPORTS["sitk"].GetImageFromArray(tensor.cpu().numpy())
    OPTIONAL_IMPORTS["sitk"].WriteImage(sitk_image, str(path), useCompression=True)


def write_sitk_nrrd(path, tensor):
    _write_sitk_image(path, tensor, suffix=".nrrd")


def write_seg_nrrd(path, tensor):
    _write_sitk_image(path, tensor, suffix=".seg.nrrd")


def write_sitk_nifti(path, tensor):
    _write_sitk_image(path, tensor, suffix=".nii.gz")
