import sys

import torch
import torchvision
from loguru import logger
from monai.utils.module import optional_import

from lighter.callbacks.utils import preprocess_image
from lighter.callbacks.writer.base import LighterBaseWriter
from lighter.utils.dynamic_imports import OPTIONAL_IMPORTS


class LighterFileWriter(LighterBaseWriter):
    def write(self, idx, identifier, tensor, write_format):
        filename = f"{write_format}" if identifier is None else f"{identifier}_{write_format}"
        write_dir = self.write_dir / str(idx)
        write_dir.mkdir()

        if write_format is None:
            pass
        elif write_format == "tensor":
            path = write_dir / f"{filename}.pt"
            torch.save(tensor, path)
        elif write_format == "image":
            path = write_dir / f"{filename}.png"
            torchvision.io.write_png(preprocess_image(tensor), path)
        elif write_format == "video":
            path = write_dir / f"{filename}.mp4"
            torchvision.io.write_video(path, tensor, fps=24)
        elif write_format == "scalar":
            raise NotImplementedError
        elif write_format == "audio":
            raise NotImplementedError
        elif write_format in ["nifti", "nrrd"]:
            ext = "nii.gz" if write_format == "nifti" else "nrrd"
            path = write_dir / f"image.{ext}"
            write_sitk_image(path, tensor)
        else:
            logger.error(f"`write_format` '{write_format}' not supported.")
            sys.exit()


def write_sitk_image(path: str, tensor: torch.Tensor) -> None:
    """Write a SimpleITK image to disk.

    Args:
        path (str): path to write the image.
        tensor (torch.Tensor): tensor to write.
    """
    if "sitk" not in OPTIONAL_IMPORTS:
        OPTIONAL_IMPORTS["sitk"], sitk_available = optional_import("SimpleITK")
        if not sitk_available:
            raise ModuleNotFoundError("SimpleITK not installed. To install it, run `pip install SimpleITK`. Exiting.")
    sitk_image = OPTIONAL_IMPORTS["sitk"].GetImageFromArray(tensor.cpu().numpy())
    OPTIONAL_IMPORTS["sitk"].WriteImage(sitk_image, str(path), True)
