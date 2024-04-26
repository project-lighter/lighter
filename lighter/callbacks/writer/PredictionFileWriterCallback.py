from typing import Any, Protocol, Sequence

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partialmethod
from operator import itemgetter
from pathlib import Path

import ffmpeg
import numpy as np
import torch
from einops import rearrange, repeat
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from monai.data import MetaTensor, decollate_batch
from monai.utils import convert_to_cupy, convert_to_tensor, optional_import

from lighter import LighterSystem
from lighter.defs import ModeEnum
from lighter.postprocessing.pipeline import ProcessingPipeline
from lighter.utils.color import distinguishable_colors
from lighter.utils.json import TensorEncoder


class PredictionFormatter(Protocol):
    def __call__(self, *, image: torch.Tensor | None = None, label: torch.Tensor | None) -> torch.Tensor:
        pass


@dataclass(kw_only=True)
class BaseRGBPredictionFormatter(PredictionFormatter):
    image_channel: int | None = None
    label_one_hot: bool = False

    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        min_v = image.min()
        max_v = image.max()
        return (((image - min_v) / (max_v - min_v)) * 255).round().to(dtype=torch.uint8)

    @torch.no_grad()
    def convert_to_rgb_tensor(self, image: torch.Tensor, label: torch.Tensor | None = None) -> torch.Tensor:
        assert image.ndim == 4
        if self.image_channel is not None:
            image = image[self.image_channel, None, ...]
        if image.shape[0] != 1:
            raise ValueError(
                f"Image should have a single channel, got {image.shape[0]}, use image_channel"
                f" to specify the channel to use"
            )
        result = repeat(self.normalize_image(image), "c x y z -> z y x ( repeat c )", repeat=3)
        if label is not None:
            if self.label_one_hot:
                n_classes = label.shape[0]
                colors = np.concatenate(
                    [np.zeros((1, 3), dtype=np.uint8), distinguishable_colors(n_classes, return_as_uint8=True)]
                )
                colors = torch.tensor(colors).to(dtype=torch.uint8)
                label = rearrange(label, "c x y z -> c z y x")
                for i in range(n_classes):
                    edges = self.find_edges(label[i, ...])
                    label = label.to(device=edges.device)
                    result = result.to(device=edges.device)
                    colors = colors.to(device=edges.device)
                    result[edges] = colors[i]

            elif n_classes := label.max().item():
                if label.shape[0] != 1:
                    raise ValueError("Label should have a single channel if not one-hot encoded")
                label = label[0, ...].to(dtype=torch.int)
                label = rearrange(label, "x y z -> z y x")
                n_classes = int(n_classes)
                edges = self.find_edges(label)
                label = label.to(device=edges.device)
                result = result.to(device=edges.device)
                # 0 is background, so add a placeholder color for it, it won't be used in the final image because of where
                colors = np.concatenate(
                    [np.zeros((1, 3), dtype=np.uint8), distinguishable_colors(n_classes, return_as_uint8=True)]
                )
                colors = torch.tensor(colors).to(device=edges.device, dtype=torch.uint8)
                result.masked_scatter_(edges.unsqueeze(-1), colors[label])
        return result.cpu()

    def find_edges(self, label):
        edges: torch.Tensor | None = None
        boundary_mode = "inner"
        if label.device.type == "cuda":
            find_boundaries_cu, has_cucim = optional_import("cucim.skimage.segmentation", name="find_boundaries")
            cp, has_cp = optional_import("cupy")

            if has_cucim and has_cp:
                prediction_cp = convert_to_cupy(label)

                edges = convert_to_tensor(find_boundaries_cu(prediction_cp, mode=boundary_mode))
        if edges is None:
            find_boundaries_skimage, _ = optional_import("skimage.segmentation", name="find_boundaries")
            edges = convert_to_tensor(find_boundaries_skimage(label.cpu().numpy(), mode=boundary_mode))
        return edges

    def __call__(self, *, image: torch.Tensor | None = None, label: torch.Tensor | None) -> torch.Tensor:
        return self.convert_to_rgb_tensor(image, label)


class PredictionWriter(Protocol):
    def __call__(
        self,
        *,
        formated_tensor: torch.Tensor,
        sample: dict[str, Any],
        trainer: Trainer,
        system: LighterSystem,
        mode: ModeEnum,
        batch_idx: int,
        sample_idx: int,
    ) -> None:
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

    def get_metadata(
        self, *, formated_tensor: torch.Tensor, sample: dict[str, Any], trainer: Trainer, system: LighterSystem, mode: ModeEnum
    ) -> dict[str, Any]:
        metadata = {}
        extracted_keys = ["input", "label", "prediction", "pred"]
        for key in extracted_keys:
            match (sample.get(key, None)):
                case MetaTensor() as image:
                    instance_meta = {}
                    instance_meta["applied_operations"] = image.applied_operations
                    # FIXME Tensors are not serializable, need to convert to list
                    # instance_meta["meta"] = image.meta
                    metadata[key] = instance_meta

        # Fixme make flexible to support other keys

        return metadata

    def __call__(
        self,
        *,
        formated_tensor: torch.Tensor,
        sample: dict[str, Any],
        trainer: Trainer,
        system: LighterSystem,
        mode: ModeEnum,
        batch_idx: int,
        sample_idx: int,
    ) -> None:
        template_data = {
            "mode": mode,
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "batch_idx": batch_idx,
            "sample_idx": sample_idx,
            "id": sample["id"],
        }
        output_path = self.output_root / Path(self.file_template.format(**template_data))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_metadata and (
            metadata := self.get_metadata(
                formated_tensor=formated_tensor, sample=sample, trainer=trainer, system=system, mode=mode
            )
        ):
            metadata_path = output_path.with_suffix(".json")
            with metadata_path.open("w") as f:
                json.dump(metadata, f, cls=TensorEncoder, indent=2)
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


class TensorExtractor(Protocol):
    def __call__(self, data: dict[str, Any]) -> torch.Tensor:
        pass


class PredictionWriterCallback(Callback):
    def __init__(
        self,
        modes: Sequence[ModeEnum],
        formatter: PredictionFormatter,
        writer: PredictionWriter,
        max_predictions: int | None = None,
        processing_step: str | None = None,
        every_n_epochs: int | None = None,
        write_on_sanity_check: bool = False,
        image_extractor: TensorExtractor | str = "input",
        label_extractor: TensorExtractor | str = "prediction",
        **kwargs,
    ):
        self.written_counts = {}
        self.modes = {ModeEnum(mode) for mode in modes}
        self.max_predictions = max_predictions
        self.processing_step = processing_step
        self.formatter = formatter
        self.writer = writer
        self.every_n_epochs = every_n_epochs
        self.write_on_sanity_check = write_on_sanity_check

        self.image_extractor = self._init_extractor(image_extractor)
        self.label_extractor = self._init_extractor(label_extractor)

    @staticmethod
    def _init_extractor(extractor: TensorExtractor | str) -> TensorExtractor:
        if extractor is None:
            return lambda x: x
        if isinstance(extractor, str):
            return itemgetter(extractor)
        return extractor

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

    def handle_epoch_end(self, mode: ModeEnum, trainer: Trainer, system: LighterSystem) -> None:
        self.reset_written_counts(mode)

    @torch.no_grad()
    def handle_batch(
        self,
        mode: ModeEnum,
        trainer: Trainer,
        system: LighterSystem,
        outputs: STEP_OUTPUT,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        if mode not in self.modes:
            return
        if not self.write_on_sanity_check and trainer.sanity_checking:
            return
        if self.every_n_epochs is not None and trainer.current_epoch % self.every_n_epochs != 0:
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
            data = {k: v for k, v in outputs.items() if k in ["input", "pred"]}
        else:
            pipeline: ProcessingPipeline = outputs["pipeline"]
            data = pipeline.get_result(self.processing_step)
        for i, sample in enumerate(decollate_batch(data)):
            if self.max_predictions is not None:
                if mode not in self.written_counts:
                    self.written_counts[mode] = 0
                if self.written_counts[mode] >= self.max_predictions:
                    return
                self.written_counts[mode] += 1
            formated_tensor = self.formatter(image=sample.get("input", None), label=sample.get("pred", None))
            self.writer(
                formated_tensor=formated_tensor,
                sample=sample,
                trainer=trainer,
                system=system,
                mode=mode,
                batch_idx=batch_idx,
                sample_idx=i,
            )

    on_train_batch_end = partialmethod(handle_batch, ModeEnum.TRAIN)
    on_validation_batch_end = partialmethod(handle_batch, ModeEnum.VAL)
    on_predict_batch_end = partialmethod(handle_batch, ModeEnum.PREDICT)
    on_test_batch_end = partialmethod(handle_batch, ModeEnum.TEST)

    on_train_epoch_end = partialmethod(handle_epoch_end, ModeEnum.TRAIN)
    on_validation_epoch_end = partialmethod(handle_epoch_end, ModeEnum.VAL)
    on_predict_epoch_end = partialmethod(handle_epoch_end, ModeEnum.PREDICT)
    on_test_epoch_end = partialmethod(handle_epoch_end, ModeEnum.TEST)
