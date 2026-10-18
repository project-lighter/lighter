"""Callback for persisting prediction artifacts to the filesystem."""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import LightningModule, Trainer

from lighter.callbacks.base_writer import BaseWriter

logger = logging.getLogger(__name__)


def write_tensor(path: Path, tensor: torch.Tensor, *, suffix: str = ".pt") -> None:
    """Serialise a tensor to disk using :func:`torch.save`."""

    torch.save(tensor, path.with_suffix(suffix))  # nosec B614


def _write_png(tensor: torch.Tensor, path: Path) -> None:
    """Scale a CHW tensor from [0, 1] to uint8 and save it as PNG."""
    try:
        from torchvision.io import write_png
    except ImportError as e:
        raise ImportError("The image writers need torchvision: pip install torchvision") from e
    write_png((tensor.float().clamp(0, 1) * 255).to(torch.uint8), str(path))


def write_image_2d(path: Path, tensor: torch.Tensor, *, suffix: str = ".png") -> None:
    """Write a 2D tensor as an image using PNG encoding."""
    if tensor.ndim != 3:
        raise ValueError(f"write_image_2d expects a 3D tensor (CHW), got {tensor.ndim} dimensions.")
    _write_png(tensor, path.with_suffix(suffix))


def write_image_3d(path: Path, tensor: torch.Tensor, *, suffix: str = ".png") -> None:
    """Write a 3D tensor as a 2D image by stacking slices vertically."""
    if tensor.ndim != 4:
        raise ValueError(f"write_image_3d expects a 4D tensor (CDHW), got {tensor.ndim} dimensions.")
    # CDHW -> C(D*H)W
    shape = tensor.shape
    _write_png(tensor.view(shape[0], shape[1] * shape[2], shape[3]), path.with_suffix(suffix))


def write_text(path: Path, value: Any, *, suffix: str = ".txt", encoding: str = "utf-8") -> None:
    """Write the string representation of *value* to disk."""

    path = path.with_suffix(suffix)
    with path.open("w", encoding=encoding) as file:
        file.write(str(value))


WRITERS: dict[str, Callable[..., None]] = {
    "tensor": write_tensor,
    "image_2d": write_image_2d,
    "image_3d": write_image_3d,
    "text": write_text,
}


class FileWriter(BaseWriter):
    """
    Persist a prediction value per sample to disk.

    Args:
        directory: Directory to save prediction files.
        value_key: Key in the prediction outputs dict containing values to save.
        writer_fn: Name of a writer in `WRITERS` ("tensor", "image_2d", "image_3d", "text") or a callable
            taking `(path, value)`.
        name_key: Optional key for custom file names. If None, uses sequential numbering.

    Example:
        ```yaml
        trainer:
          callbacks:
            - _target_: lighter.callbacks.FileWriter
              directory: predictions/
              value_key: pred
              writer_fn: tensor
        ```
    """

    def __init__(
        self,
        directory: str | Path,
        value_key: str,
        writer_fn: str | Callable[[Path, Any], None],
        name_key: str | None = None,
    ) -> None:
        super().__init__(directory)
        self.value_key = value_key
        self.name_key = name_key
        if isinstance(writer_fn, str):
            if writer_fn not in WRITERS:
                raise ValueError(f"Unknown writer_fn '{writer_fn}'. Use one of {sorted(WRITERS)} or a callable.")
            self.writer_fn = WRITERS[writer_fn]
        elif callable(writer_fn):
            self.writer_fn = writer_fn
        else:
            raise TypeError("writer_fn must be a string or a callable")

        self._counter: int | None = None
        self._step: int = 1

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if stage != "predict":
            return

        if self.path.suffix:
            raise ValueError("FileWriter expects 'directory' to be a directory path, not a file path")

        if trainer.is_global_zero:
            self.path.mkdir(parents=True, exist_ok=True)

        if trainer.world_size > 1:
            self._step = trainer.world_size
            self._counter = trainer.global_rank
        else:
            self._step = 1
            self._counter = 0

    def write(self, outputs: dict[str, Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:  # noqa: ARG002
        if self._counter is None:
            logger.debug("FileWriter received outputs before setup; skipping batch")
            return

        values = self._to_sequence(outputs, self.value_key)
        if not values:
            logger.debug(f"FileWriter value key '{self.value_key}' yielded no samples; skipping batch")
            return

        if self.name_key is not None:
            names = self._to_sequence(outputs, self.name_key)
            if len(names) != len(values):
                raise ValueError(
                    "Length mismatch between value key "
                    f"'{self.value_key}' ({len(values)}) and name key "
                    f"'{self.name_key}' ({len(names)})."
                )
        else:
            names = []

        for offset, value in enumerate(values):
            global_index = self._counter + offset * self._step
            name = self._prepare_name(names[offset]) if names else global_index

            target_path = self.path / str(name)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            prepared_value = self._prepare_value(value)
            self.writer_fn(target_path, prepared_value)

        self._counter += len(values) * self._step

    @staticmethod
    def _prepare_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return value

    @staticmethod
    def _prepare_name(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist()
        return value

    @staticmethod
    def _to_sequence(outputs: dict[str, Any], key: str) -> list:
        if key not in outputs:
            raise KeyError(f"FileWriter expected key '{key}' in outputs but it was missing.")

        value = outputs[key]
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return [value]
            return [tensor for tensor in value]
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
        return [value]
