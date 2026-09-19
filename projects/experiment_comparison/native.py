"""The same experiment expressed directly with native Lightning APIs."""

import argparse
import csv
import json
import sys
import time
import uuid
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

if __package__:
    from .artifacts import ExperimentArtifacts, write_json
    from .data import ComparisonData
    from .task import PREDICTION_COLUMNS, Classifier, ScientificSteps
else:
    from artifacts import ExperimentArtifacts, write_json
    from data import ComparisonData
    from task import PREDICTION_COLUMNS, Classifier, ScientificSteps


class NativeClassifier(ScientificSteps, pl.LightningModule):
    def __init__(self, lr=0.01, initial_state=None):
        super().__init__()
        self.network = Classifier(initial_state=initial_state)
        self.lr = lr

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.network.parameters(), lr=self.lr, momentum=0.9, dampening=0, weight_decay=0, nesterov=False
        )


class NativeCsvWriter(pl.Callback):
    """The native baseline writes one prediction batch at a time."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def on_predict_start(self, trainer, pl_module):
        self.temporary = self.path.with_suffix(".pending.csv")
        self.stream = self.temporary.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=PREDICTION_COLUMNS)
        self.writer.writeheader()

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        for index in range(len(outputs["id"])):
            self.writer.writerow(
                {key: values[index].item() if torch.is_tensor(values) else values[index] for key, values in outputs.items()}
            )

    def on_predict_end(self, trainer, pl_module):
        self.stream.close()
        self.temporary.replace(self.path)

    def teardown(self, trainer, pl_module, stage):
        if self.stream is not None:
            self.stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "validate", "test", "predict"))
    parser.add_argument("--data-manifest", type=Path, default=Path("data/data.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--order-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--accelerator", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--precision", choices=("32-true", "16-mixed", "bf16-mixed"), default="32-true")
    parser.add_argument("--ckpt-path", type=Path)
    parser.add_argument("--parent-attempt-id")
    parser.add_argument("--trace-updates", action="store_true")
    options = parser.parse_args(argv)
    options.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    attempt = {
        "attempt_id": str(uuid.uuid4()),
        "parent_attempt_id": options.parent_attempt_id,
        "stage": options.stage,
        "requested_lr": options.lr,
        "requested_accelerator": options.accelerator,
        "requested_precision": options.precision,
        "checkpoint": str(options.ckpt_path) if options.ckpt_path else None,
        "argv": sys.argv,
        "status": "running",
    }
    write_json(options.output_dir / "native-attempt.json", attempt)
    torch.set_num_threads(2)
    pl.seed_everything(options.seed, workers=True)
    # Restored weights have explicit precedence. The supplied initial artifact
    # remains recorded by workflow.py but is not used to initialize this model.
    model = NativeClassifier(options.lr, None if options.ckpt_path else options.initial_state)
    data = ComparisonData(options.data_manifest, options.order_manifest, seed=options.seed, batch_size=options.batch_size)
    checkpoint = ModelCheckpoint(
        dirpath=options.output_dir / "checkpoints",
        filename="epoch-{epoch:02d}-step-{step}",
        auto_insert_metric_name=False,
        monitor="val/ce",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator=options.accelerator,
        devices=1,
        precision=options.precision,
        max_epochs=options.max_epochs,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        deterministic=True,
        default_root_dir=options.output_dir,
        callbacks=[
            checkpoint,
            ExperimentArtifacts(options.output_dir, options.trace_updates, requested_lr=options.lr),
            NativeCsvWriter(options.output_dir / "predictions.csv"),
        ],
    )
    kwargs = {"model": model, "datamodule": data, "ckpt_path": options.ckpt_path}
    if options.stage == "predict":
        kwargs["return_predictions"] = False
    if options.stage in ("validate", "test"):
        kwargs["verbose"] = False
    try:
        getattr(trainer, options.stage)(**kwargs)
    except Exception as error:
        attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        attempt["status"] = "completed"
    finally:
        attempt["elapsed_seconds"] = time.monotonic() - started
        write_json(options.output_dir / "native-attempt.json", attempt)
    print(json.dumps(attempt))


if __name__ == "__main__":
    main()
