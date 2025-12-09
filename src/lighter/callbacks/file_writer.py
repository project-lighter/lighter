"""Callback for persisting prediction artifacts to the filesystem."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torchvision
from loguru import logger
from pytorch_lightning import Trainer

from lighter.callbacks.base_writer import BaseWriter
from lighter.model import LighterModule
from lighter.utils.types.enums import Stage

#
# Registry
#


class WriterRegistry:
    """A registry for writer functions, allowing them to be registered by name and retrieved later."""

    def __init__(self) -> None:
        self._registry: dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
        """Register a new writer function in this registry as a decorator.

        Args:
            name: The unique name to register the writer under.

        Returns:
            A decorator that registers the decorated function.

        Raises:
            ValueError: If a writer with the given name is already registered.
        """

        def decorator(fn: Callable) -> Callable:
            if name in self._registry:
                raise ValueError(f"Writer with name '{name}' is already registered.")
            self._registry[name] = fn
            return fn

        return decorator

    def get(self, name: str) -> Callable:
        """Get a writer from the registry by its registered name.

        Args:
            name: The name of the writer to retrieve.

        Returns:
            The writer function associated with the given name.

        Raises:
            ValueError: If no writer with the given name is registered.
        """
        if name not in self._registry:
            raise ValueError(f"Writer with name '{name}' is not registered.")
        return self._registry[name]


writer_registry = WriterRegistry()


#
# Writer Functions
#
@writer_registry.register(name="tensor")
def write_tensor(path: Path, tensor: torch.Tensor, *, suffix: str = ".pt") -> None:
    """Serialise a tensor to disk using :func:`torch.save`."""

    torch.save(tensor, path.with_suffix(suffix))  # nosec B614


@writer_registry.register(name="image_2d")
def write_image_2d(path: Path, tensor: torch.Tensor, *, suffix: str = ".png") -> None:
    """Write a 2D tensor as an image using PNG encoding."""
    if tensor.ndim != 3:
        raise ValueError(f"write_image_2d expects a 3D tensor (CHW), got {tensor.ndim} dimensions.")
    path = path.with_suffix(suffix)
    # Scale to [0, 255] and convert to uint8
    tensor = (tensor.float().clamp(0, 1) * 255).to(torch.uint8)
    torchvision.io.write_png(tensor, str(path))


@writer_registry.register(name="image_3d")
def write_image_3d(path: Path, tensor: torch.Tensor, *, suffix: str = ".png") -> None:
    """Write a 3D tensor as a 2D image by stacking slices vertically."""
    if tensor.ndim != 4:
        raise ValueError(f"write_image_3d expects a 4D tensor (CDHW), got {tensor.ndim} dimensions.")
    path = path.with_suffix(suffix)
    # CDHW -> C(D*H)W
    shape = tensor.shape
    tensor = tensor.view(shape[0], shape[1] * shape[2], shape[3])
    # Scale to [0, 255] and convert to uint8
    tensor = (tensor.float().clamp(0, 1) * 255).to(torch.uint8)
    torchvision.io.write_png(tensor, str(path))


@writer_registry.register(name="text")
def write_text(path: Path, value: Any, *, suffix: str = ".txt", encoding: str = "utf-8") -> None:
    """Write the string representation of *value* to disk."""

    path = path.with_suffix(suffix)
    with path.open("w", encoding=encoding) as file:
        file.write(str(value))


class FileWriter(BaseWriter):
    """
    Persist a prediction value per sample to disk.

    Args:
        directory: Directory to save prediction files.
        value_key: Key in the prediction outputs dict containing values to save.
        writer_fn: Writer function name (e.g., "tensor", "image_2d", "text") or callable.
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
            self.writer_fn = writer_registry.get(writer_fn)
        elif callable(writer_fn):
            self.writer_fn = writer_fn
        else:
            raise TypeError("writer_fn must be a string or a callable")

        self._counter: int | None = None
        self._step: int = 1

    def setup(self, trainer: Trainer, pl_module: LighterModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if stage != Stage.PREDICT:
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
            logger.debug("FileWriter value key '{}' yielded no samples; skipping batch", self.value_key)
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
