import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partialmethod
from io import RawIOBase
from pathlib import Path
from typing import Any, Sequence, Protocol, Callable

import torch
from monai.data import decollate_batch, MetaTensor
from monai.utils import optional_import, convert_to_cupy, convert_to_tensor
from pytorch_lightning import Trainer, Callback
from torch.nn import Embedding

from lighter import LighterSystem
from lighter.callbacks.writer.file import LighterFileWriter
from enum import StrEnum
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from lighter.defs import ModeEnum
from lighter.postprocessing.pipeline import ProcessingPipeline
from einops import repeat

from lighter.utils.color import distinguishable_colors
import ffmpeg

class PredictionFormatter(Protocol):
    def __call__(self,*, image: torch.Tensor | None = None, prediction: torch.Tensor | None ) -> torch.Tensor:
        pass

class BaseRGBPredictionFormatter(PredictionFormatter):


    def normalize_image(self, image:torch.Tensor) -> torch.Tensor:
        min_v = image.min()
        max_v = image.max()
        return (((image - min_v) / (max_v-min_v)) * 255).round().to(dtype=torch.uint8)

    @torch.no_grad()
    def convert_to_rgb_tensor(self, image: torch.Tensor, prediction: torch.Tensor | None = None) -> torch.Tensor:
        assert image.ndim == 4
        assert image.shape[0] == 1
        result = repeat(self.normalize_image(image), "c z y x -> z y x c", c=3)
        if prediction is not None and (n_classes:= prediction.max().item()):

            edges: torch.Tensor | None = None
            if prediction.device.type == "cuda":
                find_boundaries_cu, has_cucim = optional_import(
                    "cucim.skimage.segmentation", name="find_boundaries"
                )
                cp, has_cp = optional_import("cupy")

                if has_cucim and has_cp:
                    prediction_cp = convert_to_cupy(prediction)
                    edges = find_boundaries_cu(prediction_cp, mode="outer")
            if edges is None:
                find_boundaries_skimage, _ = optional_import("skimage.segmentation", name="find_boundaries")
                edges = convert_to_tensor(find_boundaries_skimage(prediction.cpu().numpy(), mode="outer"))
            prediction = prediction.to(device=edges.device)
            result = result.to(device=edges.device)
            colors = distinguishable_colors(n_classes, return_as_uint8=True)
            color_embedding = Embedding(num_embeddings=n_classes, embedding_dim=3, _weight=torch.tensor(colors)).to(edges.device)
            result = torch.where(edges.unsqueeze(-1), color_embedding(prediction), result)
        return result.cpu()

    def __call__(self, *, image: torch.Tensor | None = None, prediction: torch.Tensor | None) -> torch.Tensor:
        return self.convert_to_rgb_tensor(image, prediction)


class PredictionWriter(Protocol):
    def __call__(self, *, formated_tensor: torch.Tensor, sample: dict[str,Any], trainer: Trainer, system: LighterSystem,mode:ModeEnum, batch_idx: int, sample_idx:int) -> None:
        pass

@dataclass(kw_only=True)
class BasePredictionFileWriter(PredictionWriter, ABC):
    output_root: Path
    log_metadata: bool = True
    file_template: str = "{mode}/{epoch}/{id}_{batch_idx}_{sample_idx}"
    def __post_init__(self):
        self.output_root = Path(self.output_root)

    @abstractmethod
    def write(self, tensor: torch.Tensor, file_base: Path) -> None:
        raise NotImplemented
    def get_metadata(self, *, formated_tensor: torch.Tensor, sample: dict[str,Any], trainer: Trainer, system: LighterSystem, mode:ModeEnum) -> dict[str,Any]:
        metadata = {}

        if (image := sample.get("image",None)) and isinstance(image, MetaTensor):
            metadata["image"] = image.meta
        if (prediction := sample.get("prediction",None)) and isinstance(prediction, MetaTensor):
            metadata["prediction"] = prediction.meta
        return metadata
    def __call__(self, *, formated_tensor: torch.Tensor, sample: dict[str,Any], trainer: Trainer, system: LighterSystem,mode:ModeEnum, batch_idx: int, sample_idx:int) -> None:
        template_data = {
            "mode": mode,
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "batch_idx": batch_idx,
            "sample_idx": sample_idx,
            "id": sample["id"]
        }
        output_path = self.output_root / Path(self.file_template.format(**sample))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_metadata and (metadata := self.get_metadata(formated_tensor=formated_tensor, sample=sample, trainer=trainer, system=system, mode=mode)):
            metadata_path = output_path.with_suffix(".json")
            with metadata_path.open("w") as f:
                json.dump(metadata, f)
        self.write(tensor=formated_tensor, file_base=output_path)

@dataclass(kw_only=True)
class PredictionVideoWriter(BasePredictionFileWriter):
    fps: int = 30

    def write(self, tensor: torch.Tensor, file_base: Path) -> None:
        output_path = file_base.with_suffix(".mp4")
        out, _ = (
            ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s="{}x{}".format(*tensor.shape[1:3]), r=self.fps)
            .output(output_path.as_posix(), pix_fmt="yuv420p", vcodec="libx264", crf=25)
            .overwrite_output()
            .run(quiet=True, input=tensor.numpy().tobytes())
        )

class PredictionWriterCallback(Callback):
    def __init__(self, modes: Sequence[ModeEnum], formatter: PredictionFormatter, writer: PredictionWriter, max_predictions: int | None = None, processing_step: str | None = None,
                  **kwargs):
        super().__init__(**kwargs)
        self.written_counts = {}
        self.modes = set((ModeEnum(mode) for mode in modes))
        self.max_predictions = max_predictions
        self.processing_step = processing_step
        self.formatter = formatter
        self.writer = writer
    def get_remaining_predictions(self, mode: ModeEnum) -> int | None:
        if self.max_predictions is None:
            return
        return self.max_predictions - self.written_counts.get(mode, 0)

    def update_written_counts(self, mode: ModeEnum, count: int) -> None:
        if self.max_predictions is None:
            return
        if mode not in self.written_counts:
            self.written_counts[mode] = 0
        self.written_counts[mode] += count

    def reset_written_counts(self, mode: ModeEnum) -> None:
        self.written_counts[mode] = 0



    @torch.no_grad()
    def handle_batch(self,mode: ModeEnum,trainer: pl.Trainer, system: LighterSystem, outputs: STEP_OUTPUT, batch: dict[str,Any], batch_idx: int) -> None:
        if mode not in self.modes:
            return

        if self.get_remaining_predictions(mode) < 0:
            return
        # If the IDs are not provided, generate global unique IDs based on the prediction count. DDP supported.
        # if outputs["id"] is None:
        #     batch_size = len(outputs["pred"])
        #     world_size = trainer.world_size
        #     outputs["id"] = list(range(self._pred_counter, self._pred_counter + batch_size * world_size, world_size))
        #     self._pred_counter += batch_size * world_size
        if self.processing_step is None:
            data = {k:v for k,v in batch.items() if k in ["input", "pred"]}
        else:
            pipeline: ProcessingPipeline = batch["pipeline"]
            data = pipeline.get_result(self.processing_step)
        for i, sample in enumerate(decollate_batch(data)):
            if self.max_predictions is not None:
                if mode not in self.written_counts:
                    self.written_counts[mode] = 0
                if self.written_counts[mode] >= self.max_predictions:
                    return
                self.written_counts[mode] += 1
            formated_tensor = self.formatter(image=sample.get("input", None), prediction=sample.get("pred",None))
            self.writer(formated_tensor=formated_tensor, sample=sample, trainer=trainer, system=system, mode=mode, batch_idx=batch_idx, sample_idx=i)

    on_train_batch_end = partialmethod(handle_batch, ModeEnum.TRAIN)
    on_validation_batch_end = partialmethod(handle_batch, ModeEnum.VAL)
    on_predict_batch_end = partialmethod(handle_batch, ModeEnum.PREDICT)
    on_test_batch_end = partialmethod(handle_batch, ModeEnum.TEST)


