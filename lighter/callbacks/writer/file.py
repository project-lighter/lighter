import sys

import torch
import torchvision
from loguru import logger
from monai.transforms import DivisiblePad
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

        # Tensor
        elif write_format == "tensor":
            path = write_dir / f"{filename}.pt"
            torch.save(tensor, path)

        # Image
        elif write_format == "image":
            path = write_dir / f"{filename}.png"
            torchvision.io.write_png(preprocess_image(tensor), path)

        # Video
        elif write_format == "video":
            path = write_dir / f"{filename}.mp4"
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

        # Scalar
        elif write_format == "scalar":
            raise NotImplementedError

        # Audio
        elif write_format == "audio":
            raise NotImplementedError

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
