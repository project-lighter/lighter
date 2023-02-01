import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import itertools
from pathlib import Path
from datetime import datetime

from loguru import logger
import torch
import torchvision
from pytorch_lightning import Callback, Trainer

from lighter import LighterSystem
from lighter.callbacks.utils import LIGHTNING_TO_LIGHTER_STAGE, parse_data, concatenate, preprocess_image


class LighterWriter(Callback):
    def __init__(self, write_dir: str, write_as: str, write_on: str = "step", write_to_csv: bool = False) -> None:
        self.write_dir = Path(write_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.write_as = write_as
        self.write_on = write_on
        self.write_to_csv = write_to_csv

    def setup(self, trainer: Trainer, pl_module: LighterSystem, stage: str) -> None:
        if self.write_on not in ["step", "epoch"]:
            logger.error("`write_on` must be either 'step' or 'epoch'.")
            sys.exit()

        if self.write_to_csv and self.write_as in ["image", "tensor"]:
            logger.error(f"`write_as={self.write_as}` cannot be written to a CSV. Change `write_as` or disable `write_to_csv`.")
            sys.exit()

        self.write_dir.mkdir(parents=True)

    def _write(self, outputs, indices):
        for identifier, data in parse_data(outputs):
            for idx, tensor in zip(indices, data):
                name = f"step_{idx}" if identifier is None else  f"step_{idx}_{identifier}"
                if self.write_as == "tensor":
                    path = self.write_dir / f"{self.write_as}_{name}.pt"
                    torch.save(tensor, path)
                elif self.write_as == "image":
                    path = self.write_dir / f"{self.write_as}_{name}.png"
                    torchvision.utils.save_image(preprocess_image(tensor), path)
                elif self.write_as == "scalar":
                    raise NotImplementedError
                elif self.write_as == "audio":
                    raise NotImplementedError
                elif self.write_as == "video":
                    raise NotImplementedError
                else:
                    logger.error(f"`write_as` does not support '{self.write_as}'.")
                    sys.exit()

    def on_predict_batch_end(self, trainer: Trainer, pl_module: LighterSystem, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if self.write_on != "step":
            return
        indices = trainer.predict_loop.epoch_loop.current_batch_indices
        self._write(outputs, indices)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LighterSystem, outputs: List[Any]) -> None:
        if self.write_on != "epoch":
            return

        # Only one epoch when predicting, index the lists of outputs and batch indices accordingly.
        indices = trainer.predict_loop.epoch_batch_indices[0]
        outputs = outputs[0]

        # Concatenate/flatten into a list of indices.
        indices = list(itertools.chain(*indices))
        # Concatenate/flatten the outputs so that each output corresponds to its index in `indices`.
        outputs = concatenate(outputs)

        self._write(outputs, indices)
    
    def on_predict_end(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        # Dump the CSV
        pass

