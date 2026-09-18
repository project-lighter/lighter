"""Small local evidence records around native Lightning execution.

This is an attempt log, not a scheduler, process monitor, or replacement tracker.
"""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def describe(value: Any) -> Any:
    """Describe values without invoking user repr/serialization or copying tensors."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"__nonfinite_float__": str(value)}
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: describe(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [describe(item) for item in value]
    return {"__opaque__": f"{type(value).__module__}.{type(value).__qualname__}", "replayable": False}


def _has_opaque(value: Any) -> bool:
    if isinstance(value, dict):
        return "__opaque__" in value or any(_has_opaque(item) for item in value.values())
    return isinstance(value, list) and any(_has_opaque(item) for item in value)


def atomic_write(path: Path, text: str) -> None:
    """Publish one complete file, leaving a prior record intact on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".record-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _git_identity(directory: Path, *, tracked_file: Path | None = None) -> dict[str, Any] | None:
    try:

        def git(*args: str) -> str:
            return subprocess.check_output(
                ["git", "-C", str(directory), *args], stderr=subprocess.DEVNULL, text=True, timeout=3
            ).strip()

        if tracked_file is not None:
            # An installed wheel inside an ignored environment does not belong
            # to the surrounding application's source revision.
            git("ls-files", "--error-unmatch", "--", str(tracked_file.resolve()))
        return {
            "root": git("rev-parse", "--show-toplevel"),
            "directory": str(directory.resolve()),
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
        }
    except (OSError, subprocess.SubprocessError):
        return None


def _environment() -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for name in ("lighter", "sparkwheel", "torch", "pytorch-lightning", "torchmetrics", "numpy"):
        try:
            distribution_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distribution_version = None
        module = sys.modules.get(name.replace("-", "_"))
        metadata = vars(module) if module is not None else {}
        file = metadata.get("__file__")
        packages[name] = {
            "distribution_version": distribution_version,
            "loaded_version": describe(metadata.get("__version__")),
            "module_path": file,
        }
        if file and name in ("lighter", "sparkwheel"):
            packages[name]["source_git"] = _git_identity(Path(file).parent, tracked_file=Path(file))
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "project_git": _git_identity(Path.cwd()),
    }


class RunRecorder(Callback):
    """Record one native stage attempt. All ranks participate; only rank zero writes."""

    def __init__(
        self,
        *,
        source: dict[str, Any],
        stage: str,
        seed: int,
        requested_args: dict[str, Any],
        inputs: list[Any],
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.source = describe(source)
        self.stage = stage
        self.seed = seed
        self.requested_args = describe(requested_args)
        self.options = dict(options or {})
        allowed = {"root_dir", "name", "experiment_id", "parent_attempt_id"}
        unknown = self.options.keys() - allowed
        if unknown:
            raise ValueError(f"Unknown run record options: {sorted(unknown)}")
        for key, value in self.options.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"run::{key} must be a nonempty literal string")
        self.inputs = describe(inputs)
        self.input_files = []
        for item in inputs:
            if isinstance(item, (str, Path)) and "=" not in str(item):
                path = Path(item)
                if path.is_file():
                    self.input_files.append(
                        {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    )
        self.attempt_id = uuid.uuid4().hex
        self.record: dict[str, Any] = {}
        self.path: Path | None = None
        self.last_error: str | None = None
        self._training_metrics: dict[str, Any] = {}

    def _publish(self) -> None:
        if self.path is None:
            return
        try:
            atomic_write(self.path, json.dumps(self.record, indent=2, sort_keys=True, allow_nan=False) + "\n")
        except OSError as error:
            self.last_error = f"{type(error).__name__}: {error}"
            warnings.warn(
                f"Lighter could not write experiment record {self.path}: {self.last_error}", RuntimeWarning, stacklevel=2
            )

    def prepare(self, trainer: Trainer) -> None:
        """Allocate a candidate path before spawn without publishing an attempt."""
        root = Path(self.options.get("root_dir", str(Path(trainer.default_root_dir) / "lighter_runs")))
        self.path = root / self.attempt_id / "record.json"

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        attempt_id = trainer.strategy.broadcast(self.attempt_id if trainer.is_global_zero else None)
        if not isinstance(attempt_id, str):
            raise RuntimeError("The native strategy did not broadcast an experiment attempt ID")
        self.attempt_id = attempt_id
        self.prepare(trainer)
        assert self.path is not None
        if not trainer.is_global_zero:
            return
        self.record = {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "stage": self.stage,
            "status": "running",
            "started_at": _utc_now(),
            "name": self.options.get("name"),
            "experiment_id": self.options.get("experiment_id"),
            "parent_attempt_id": self.options.get("parent_attempt_id"),
            "seed": {"value": self.seed, "workers": True},
            "requested": {
                "source": self.source,
                "stage_arguments": self.requested_args,
                "inputs": self.inputs,
                "input_files": self.input_files,
            },
            "source_contains_opaque_values": _has_opaque(self.source),
            "arguments_contain_opaque_values": _has_opaque(self.requested_args),
            "environment": _environment(),
            "execution": {
                "strategy": f"{type(trainer.strategy).__module__}.{type(trainer.strategy).__qualname__}",
                "world_size": trainer.world_size,
                "precision": str(trainer.precision),
                "device": str(trainer.strategy.root_device),
            },
        }
        try:
            atomic_write(self.path.with_name("config.yaml"), yaml.safe_dump(self.source, sort_keys=False, allow_unicode=True))
        except OSError as error:
            self.last_error = f"{type(error).__name__}: {error}"
            warnings.warn(f"Lighter could not write experiment source: {self.last_error}", RuntimeWarning, stacklevel=2)
        self._publish()

    @staticmethod
    def _optimizer_state(trainer: Trainer, module: LightningModule) -> list[dict[str, Any]]:
        names = {id(parameter): name for name, parameter in module.named_parameters()}

        def value(item: Any) -> Any:
            if isinstance(item, torch.Tensor) and item.numel() == 1:
                return describe(item.detach().cpu().item())
            return describe(item)

        return [
            {
                "type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
                "groups": [
                    {
                        "settings": {key: value(item) for key, item in group.items() if type(key) is str and key != "params"},
                        "non_string_settings": [
                            {"key": describe(key), "value": value(item)} for key, item in group.items() if type(key) is not str
                        ],
                        "parameters": [names.get(id(parameter), "<external parameter>") for parameter in group["params"]],
                    }
                    for group in optimizer.param_groups
                ],
            }
            for optimizer in trainer.optimizers
        ]

    def _observe_start(self, trainer: Trainer, module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        self.record["observed_start"] = {
            "global_step": trainer.global_step,
            "epoch": trainer.current_epoch,
            "checkpoint_path": str(trainer.ckpt_path) if trainer.ckpt_path is not None else None,
            "optimizers": self._optimizer_state(trainer, module),
        }
        self._publish()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._observe_start(trainer, pl_module)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.stage == "validate":
            self._observe_start(trainer, pl_module)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._observe_start(trainer, pl_module)

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._observe_start(trainer, pl_module)

    @staticmethod
    def _collect_metrics(trainer: Trainer) -> dict[str, Any]:
        # All ranks access this before strategy teardown: uncached metrics can synchronize.
        metrics: dict[str, Any] = trainer.callback_metrics
        if not trainer.is_global_zero:
            return {}
        values = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                values[key] = (
                    describe(value.detach().cpu().item())
                    if value.numel() == 1
                    else {
                        "tensor_shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "values_omitted": True,
                    }
                )
            else:
                values[key] = describe(value)
        return values

    def _progress(self, trainer: Trainer) -> None:
        metrics = self._collect_metrics(trainer)
        if trainer.is_global_zero:
            self.record.update(
                updated_at=_utc_now(),
                metrics=metrics,
                progress={"global_step": trainer.global_step, "epoch": trainer.current_epoch},
            )
            self._publish()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._progress(trainer)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._training_metrics = self._collect_metrics(trainer)

    def _complete(self, trainer: Trainer) -> None:
        values = self._training_metrics if self.stage == "fit" else self._collect_metrics(trainer)
        if not trainer.is_global_zero:
            return
        checkpoints = [
            {
                "best_model_path": callback.best_model_path,
                "last_model_path": callback.last_model_path,
                "monitor": callback.monitor,
            }
            for callback in trainer.checkpoint_callbacks
            if isinstance(callback, ModelCheckpoint)
        ]
        from lighter.callbacks.base_writer import BaseWriter

        artifacts = [
            {"type": type(callback).__name__, "path": str(callback.path), "exists_at_end": callback.path.exists()}
            for callback in getattr(trainer, "callbacks", ())
            if self.stage == "predict" and isinstance(callback, BaseWriter)
        ]
        self.record.update(
            prediction_destinations=artifacts,
            status="completed",
            finished_at=_utc_now(),
            observed_end={"global_step": trainer.global_step, "epoch": trainer.current_epoch},
            metrics=values,
            checkpoints=checkpoints,
        )
        self._publish()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._complete(trainer)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.stage == "validate":
            self._complete(trainer)
        elif self.stage == "fit" and not trainer.sanity_checking:
            self._progress(trainer)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._complete(trainer)

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._complete(trainer)

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        # Do not access metrics or initiate collectives during asymmetric failures.
        if trainer.is_global_zero and self.record:
            self.record.update(
                status="interrupted" if isinstance(exception, KeyboardInterrupt) else "failed",
                finished_at=_utc_now(),
                error={"type": type(exception).__name__, "message": str(exception)},
            )
            self._publish()


def read_record(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if file.is_dir():
        file = file / "record.json"
    result = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise ValueError(f"Unsupported experiment record schema: {file}")
    return result


def list_records(root: str | Path) -> list[dict[str, Any]]:
    """Read published attempts. A running status does not establish process liveness."""
    return [read_record(path) for path in sorted(Path(root).glob("*/record.json"))]


def diff_records(left: str | Path, right: str | Path) -> list[dict[str, Any]]:
    """Compare recorded requests and observations without executing their recipes."""
    first, second = read_record(left), read_record(right)
    changes = []
    missing = object()

    def visit(path: str, before: Any, after: Any) -> None:
        if isinstance(before, dict) and isinstance(after, dict):
            for key in sorted(before.keys() | after.keys()):
                visit(f"{path}::{key}", before.get(key, missing), after.get(key, missing))
        elif before != after:
            changes.append(
                {
                    "path": path,
                    "before_present": before is not missing,
                    "after_present": after is not missing,
                    "before": None if before is missing else before,
                    "after": None if after is missing else after,
                }
            )

    for key in ("seed", "requested", "environment", "observed_start", "progress", "observed_end", "metrics", "execution"):
        visit(key, first.get(key, missing), second.get(key, missing))
    return changes
