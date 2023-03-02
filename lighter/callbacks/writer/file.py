import sys

import torch
import torchvision
from loguru import logger

from lighter.callbacks.utils import preprocess_image
from lighter.callbacks.writer.base import LighterBaseWriter
from lighter.utils.misc import NotSupportedError


class LighterFileWriter(LighterBaseWriter):
    def write(self, idx, identifier, tensor, write_as):
        filename = f"{write_as}" if identifier is None else f"{identifier}_{write_as}"
        write_dir = self.write_dir / str(idx)
        write_dir.mkdir()

        if write_as is None:
            pass
        elif write_as == "tensor":
            path = write_dir / f"{filename}.pt"
            torch.save(tensor, path)
        elif write_as == "image":
            path = write_dir / f"{filename}.png"
            torchvision.io.write_png(preprocess_image(tensor), path)
        elif write_as == "video":
            path = write_dir / f"{filename}.mp4"
            torchvision.io.write_video(path, tensor, fps=24)
        elif write_as == "scalar":
            raise NotImplementedError
        elif write_as == "audio":
            raise NotImplementedError
        else:
            raise NotSupportedError(f"`write_as` '{write_as}' not supported.")
