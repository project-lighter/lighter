"""Small native callbacks that retain the evidence behind this experiment."""

import csv
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def network_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.network.state_dict().items()}


def runtime_identity(trainer=None):
    names = ("lighter", "sparkwheel", "torch", "pytorch_lightning", "torchmetrics", "numpy")
    return {
        "schema": 1,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "execution": None
        if trainer is None
        else {
            "device": str(trainer.strategy.root_device),
            "precision": str(trainer.precision),
            "world_size": trainer.world_size,
            "loader_num_workers": {
                stage: [
                    loader.num_workers
                    for loader in (value if isinstance(value, (list, tuple)) else [value])
                    if hasattr(loader, "num_workers")
                ]
                for stage, value in (
                    (stage, getattr(trainer, f"{stage}_dataloader" if stage == "train" else f"{stage}_dataloaders", None))
                    for stage in ("train", "val", "test", "predict")
                )
            },
        },
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "argv": sys.argv,
        "prefix": sys.prefix,
        "executable": sys.executable,
        "packages": {
            name: {
                "loaded": name in sys.modules,
                "module_path": str(Path(sys.modules[name].__file__).resolve()) if name in sys.modules else None,
                "loaded_version": getattr(sys.modules.get(name), "__version__", None),
                "distribution_version": importlib.metadata.version(name.replace("_", "-")),
            }
            for name in ("lighter", "sparkwheel", "torch", "pytorch_lightning")
        },
        "python": sys.version,
        "modules": {
            name: str(Path(module.__file__).resolve())
            for name, module in sorted(sys.modules.items())
            if name.split(".")[0] in names and getattr(module, "__file__", None)
        },
    }


class ExperimentArtifacts(pl.Callback):
    """Record actual batch IDs and optional tensors, without changing training."""

    def __init__(self, output_dir, trace_updates=False, threads=2, requested_lr=None):
        self.output_dir = Path(output_dir)
        self.trace_updates = trace_updates
        self.threads = threads
        self.requested_lr = requested_lr
        self.epochs = []
        self.updates = []
        self.pending = None
        self.claimed = False

    def setup(self, trainer, pl_module, stage):
        self.claimed = False
        torch.set_num_threads(self.threads)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Claim the scientific stage before any checkpoint/prediction writer can
        # replace files. RunRecorder may already have created its own subfolder.
        with (self.output_dir / "stage-start.json").open("x", encoding="utf-8") as stream:
            json.dump({"stage": str(stage), "pid": os.getpid()}, stream)
        self.claimed = True
        write_json(self.output_dir / "runtime.json", runtime_identity(trainer))

    def on_train_start(self, trainer, pl_module):
        optimizer = trainer.optimizers[0]
        torch.save(network_state(pl_module), self.output_dir / "observed_initial_state.pt")
        torch.save(optimizer.state_dict(), self.output_dir / "observed_initial_optimizer.pt")
        write_json(
            self.output_dir / "observed_start.json",
            {
                "epoch_zero_based": trainer.current_epoch,
                "global_step": trainer.global_step,
                "requested_lr": self.requested_lr,
                "effective_lr": optimizer.param_groups[0]["lr"],
            },
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.pending = {
            "epoch": trainer.current_epoch + 1,
            "batch_index": batch_idx,
            "global_step_before": trainer.global_step,
            "ids": list(batch["id"]),
            "labels": batch["label"].tolist(),
        }
        if self.trace_updates:
            self.pending["before"] = network_state(pl_module)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self.trace_updates:
            self.pending["gradients"] = {
                key: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
                for key, parameter in pl_module.network.named_parameters()
            }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        observation = self.pending
        observation["global_step_after"] = trainer.global_step
        observation["lr"] = trainer.optimizers[0].param_groups[0]["lr"]
        observation["loss"] = float(outputs["loss"].detach())
        with (self.output_dir / "batches.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({k: v for k, v in observation.items() if k not in ("before", "gradients")}) + "\n")
        if self.trace_updates:
            observation["after"] = network_state(pl_module)
            observation["momentum"] = {
                key: trainer.optimizers[0].state[parameter].get("momentum_buffer", torch.tensor([])).detach().cpu().clone()
                for key, parameter in pl_module.network.named_parameters()
            }
            self.updates.append(observation)
        self.pending = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        result = pl_module.validation_result
        record = {"epoch": trainer.current_epoch + 1, "global_step": trainer.global_step, **result}
        self.epochs.append(record)
        write_json(self.output_dir / "validation.json", self.epochs)
        with (self.output_dir / "epochs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "global_step", "count", "loss", "accuracy"])
            writer.writeheader()
            writer.writerows({key: row[key] for key in writer.fieldnames} for row in self.epochs)

    def on_test_end(self, trainer, pl_module):
        write_json(self.output_dir / "test.json", pl_module.test_result)

    def on_fit_end(self, trainer, pl_module):
        optimizer = trainer.optimizers[0]
        write_json(
            self.output_dir / "observed_end.json",
            {
                "epoch_zero_based": trainer.current_epoch,
                "global_step": trainer.global_step,
                "requested_lr": self.requested_lr,
                "effective_lr": optimizer.param_groups[0]["lr"],
            },
        )
        if self.trace_updates:
            torch.save(self.updates, self.output_dir / "updates.pt")
        checkpoint = trainer.checkpoint_callback
        write_json(
            self.output_dir / "checkpoints.json",
            {
                "selected": checkpoint.best_model_path,
                "last": checkpoint.last_model_path,
                "selected_score": float(checkpoint.best_model_score),
                "metadata": {
                    name: {"epoch": saved["epoch"] + 1, "global_step": saved["global_step"]}
                    for name, saved in (
                        (name, torch.load(path, map_location="cpu", weights_only=False))
                        for name, path in (("best", checkpoint.best_model_path), ("last", checkpoint.last_model_path))
                    )
                },
            },
        )
        write_json(self.output_dir / "runtime.json", runtime_identity(trainer))

    def on_exception(self, trainer, pl_module, exception):
        if self.claimed and self.trace_updates:
            torch.save(self.updates, self.output_dir / "updates-before-failure.pt")

    def teardown(self, trainer, pl_module, stage):
        if self.claimed:
            write_json(self.output_dir / "runtime.json", runtime_identity(trainer))
