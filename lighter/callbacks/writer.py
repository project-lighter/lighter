from typing import Any, Dict, List, Optional, Tuple, Union

import itertools
import sys
from datetime import datetime
from pathlib import Path

import torch
import torchvision
from loguru import logger
from pytorch_lightning import Callback, Trainer

from lighter import LighterSystem
from lighter.callbacks.utils import concatenate, parse_data, preprocess_image


class LighterWriter(Callback):
    def __init__(
        self,
        write_dir: str,
        write_as: Union[str, List[str], Dict[str, str], Dict[str, List[str]]],
        write_on: str = "step",
        write_to_csv: bool = False,
    ) -> None:
        self.write_dir = Path(write_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.write_as = write_as
        self.write_on = write_on
        self.write_to_csv = write_to_csv

        self.parsed_write_as = None

    def setup(self, trainer: Trainer, pl_module: LighterSystem, stage: str) -> None:
        if self.write_on not in ["step", "epoch"]:
            logger.error("`write_on` must be either 'step' or 'epoch'.")
            sys.exit()

        if self.write_on != "epoch" and self.write_to_csv:
            logger.error("`write_to_csv=True` supports `write_on='epoch'` only.")
            sys.exit()

        # Broadcast the `write_dir` so that all ranks write their predictions there.
        self.write_dir = trainer.strategy.broadcast(self.write_dir)
        # Let rank 0 create the `write_dir`.
        if trainer.is_global_zero:
            self.write_dir.mkdir(parents=True)
        # If `write_dir` does not exist, the ranks are not on the same storage.
        if not self.write_dir.exists():
            logger.error(
                f"Rank {trainer.global_rank} is not on the same storage as rank 0."
                "Please run the prediction only on nodes that are on the same storage."
            )
            sys.exit()

    def _write(self, outputs, indices):
        parsed_outputs = parse_data(outputs)
        parsed_write_as = self._parse_write_as(self.write_as, parsed_outputs)
        for idx in indices:
            for identifier in parsed_outputs:
                # Unlike a list/tuple/dict of Tensors, a single Tensor has 'None' as identifier since it doesn't need one.
                name = f"step_{idx}" if identifier is None else f"step_{idx}_{identifier}"
                self._write_by_type(name, parsed_outputs[identifier], parsed_write_as[identifier])

    def _write_by_type(self, name, tensor, write_as):
        if write_as == "tensor":
            path = self.write_dir / f"{name}_{write_as}.pt"
            torch.save(tensor, path)
        elif write_as == "image":
            path = self.write_dir / f"{name}_{write_as}.png"
            torchvision.io.write_png(preprocess_image(tensor), path)
        elif write_as == "video":
            path = self.write_dir / f"{name}_{write_as}.mp4"
            torchvision.io.write_video(path, tensor, fps=24)
        elif write_as == "scalar":
            raise NotImplementedError
        elif write_as == "audio":
            raise NotImplementedError
        else:
            logger.error(f"`write_as` does not support '{write_as}'.")
            sys.exit()

    def _parse_write_as(self, write_as, parsed_outputs: Dict[str, Any]):
        if self.parsed_write_as is None:
            # If `write_as` is a string (single value), all outputs will be saved in that specified format.
            if isinstance(write_as, str):
                self.parsed_write_as = {key: write_as for key in parsed_outputs}
            # Otherwise, `write_as` needs to match the structure of the outputs in order to assign each tensor its specified type.
            else:
                self.parsed_write_as = parse_data(write_as)
                if not set(self.parsed_write_as) == set(parsed_outputs):
                    logger.error("`write_as` structure does not match the prediction's structure.")
                    sys.exit()
        return self.parsed_write_as

    def on_predict_batch_end(
        self, trainer: Trainer, pl_module: LighterSystem, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
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
        # Concatenate/flatten so that each output corresponds to its index.
        indices = list(itertools.chain(*indices))
        outputs = concatenate(outputs)
        self._write(outputs, indices)

        # Dump the CSV
