"""Small same-device controls; skipped CUDA cases are not GPU qualification.

The native-only tests can run before Lighter for independent calibration. The
RTX handoff must first require a working CUDA device, then run the CUDA cases.
Trace JSON is retained under pytest's --basetemp for independent readback.
"""

import copy
import json
import math
from pathlib import Path

import pytest
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.plugins.precision import MixedPrecision
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lighter import LighterModule, Runner

PROFILES = [
    pytest.param("cpu", "32-true", id="cpu_fp32"),
    pytest.param("cpu", "bf16-mixed", id="cpu_bf16"),
    pytest.param("mps", "32-true", id="mps_fp32"),
    pytest.param("cuda", "32-true", id="cuda_fp32"),
    pytest.param("cuda", "16-mixed", id="cuda_fp16"),
    pytest.param("cuda", "bf16-mixed", id="cuda_bf16"),
]


class Network(nn.Linear):
    def __init__(self):
        super().__init__(2, 1, bias=False)
        with torch.no_grad():
            self.weight.copy_(torch.tensor([[0.5, -0.25]]))


def loader():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    y = torch.tensor([[0.25], [-0.5], [0.5], [1.0], [-1.0]])
    return DataLoader(TensorDataset(x, y), batch_size=1, shuffle=False, num_workers=0)


def scientific_step(model, batch):
    x, y = batch
    prediction = model.network(x)
    loss = model.criterion(prediction, y)
    model.scientific_observations.append(
        {
            "x": x.detach().cpu().tolist(),
            "y": y.detach().cpu().tolist(),
            "loss": float(loss.detach()),
            "input_device": x.device.type,
            "prediction_dtype": str(prediction.dtype),
        }
    )
    return loss


class Task(LighterModule):
    def training_step(self, batch, batch_idx):
        return scientific_step(self, batch)


class NativeTask(LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.network = Network()
        self.criterion = nn.MSELoss()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        loss = scientific_step(self, batch)
        self.log("train/loss/epoch", loss, on_step=False, on_epoch=True, logger=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.5)


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    return copy.deepcopy(value)


def json_value(value):
    if isinstance(value, torch.Tensor):
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class Audit(Callback):
    """Count actual optimizer.step calls, not Lightning's attempted-step counter."""

    def __init__(self, overflow=False):
        self.overflow = overflow
        self.applied_steps = 0
        self.batches = []
        self.preclip = []
        self.hook = None

    def snapshot(self, trainer, model):
        return {
            "weight": cpu_state(model.network.weight),
            "optimizer": cpu_state(trainer.optimizers[0].state_dict()),
            "precision_state": copy.deepcopy(trainer.precision_plugin.state_dict()),
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "applied_steps": self.applied_steps,
        }

    def on_train_start(self, trainer, model):
        model.scientific_observations = []
        self.initial = self.snapshot(trainer, model)
        self.model = model
        self.device = str(model.device)
        self.precision = str(trainer.precision)
        self.managed = isinstance(model, LighterModule) and model._optimizer_binding is not None
        self.hook = trainer.optimizers[0].register_step_post_hook(self.applied)

    def applied(self, optimizer, args, kwargs):
        self.applied_steps += 1

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        self.batch_idx = batch_idx

    def on_after_backward(self, trainer, model):
        if self.overflow and trainer.current_epoch == 0 and self.batch_idx == 1:
            model.network.weight.grad.fill_(float("inf"))

    def on_before_optimizer_step(self, trainer, model, optimizer):
        # Lightning's AMP hook is after unscale, but before gradient clipping.
        self.preclip.append(
            {"epoch": trainer.current_epoch, "batch": self.batch_idx, "gradient": cpu_state(model.network.weight.grad)}
        )

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.batches.append(self.snapshot(trainer, model))

    def on_train_end(self, trainer, model):
        self.final = self.snapshot(trainer, model)
        self.loss = float(trainer.callback_metrics["train/loss/epoch"])

    def teardown(self, trainer, model, stage):
        if self.hook is not None:
            self.hook.remove()


def run_control(directory, accelerator, precision, backend, *, epochs=2, checkpoint=None, overflow=False, lr=0.125):
    """Run one native or managed candidate; independent graders may read trace.json."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    audit = Audit(overflow)
    precision_options = {"precision": precision}
    if precision == "16-mixed":
        precision_options = {
            "plugins": [
                MixedPrecision("16-mixed", "cuda", scaler=torch.amp.GradScaler("cuda", init_scale=8, growth_interval=1))
            ]
        }
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        **precision_options,
        max_epochs=epochs,
        accumulate_grad_batches=2,
        gradient_clip_val=0.25,
        gradient_clip_algorithm="norm",
        callbacks=[audit],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        deterministic=True,
        default_root_dir=directory,
    )
    if backend == "native":
        trainer.fit(NativeTask(lr), loader(), ckpt_path=checkpoint)
    else:
        recipe = {
            "model": {
                "_target_": f"{__name__}.Task",
                "network": {"_target_": f"{__name__}.Network"},
                "criterion": {"_target_": "torch.nn.MSELoss"},
                "optimizer": {
                    "_target_": "torch.optim.SGD",
                    "params": "$@model::network.parameters()",
                    "lr": lr,
                    "momentum": 0.5,
                },
            },
            "trainer": trainer,
            "run": False,
        }
        Runner().run("fit", [recipe], train_dataloaders=loader(), ckpt_path=checkpoint)
        assert audit.managed, "The Lighter control must exercise managed optimizer construction"
    saved = directory / "last.ckpt"
    trainer.save_checkpoint(saved)
    audit.checkpoint = saved
    payload = {
        "backend": backend,
        "requested_accelerator": accelerator,
        "requested_precision": precision,
        "device": audit.device,
        "precision": audit.precision,
        "overflow": overflow,
        "requested_lr": lr,
        "initial": audit.initial,
        "final": audit.final,
        "preclip": audit.preclip,
        "batches": audit.batches,
        "scientific_observations": audit.model.scientific_observations,
        "epoch_loss": audit.loss,
    }
    (directory / "trace.json").write_text(json.dumps(json_value(payload), indent=2) + "\n", encoding="utf-8")
    return audit


def require_device(accelerator, precision="32-true"):
    if accelerator == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable; this is not a CUDA qualification pass")
    if accelerator == "cuda" and precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
        pytest.skip("The available CUDA device does not support bfloat16")
    if accelerator == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS hardware unavailable")


def assert_same(actual, expected):
    # Same hardware, same precision, same arithmetic. No CPU/AMP equality claim.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, equal_nan=False)


def assert_finite(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for item in value.values():
            assert_finite(item)
    elif isinstance(value, list):
        for item in value:
            assert_finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


def check_lifecycle(directory, accelerator, precision, backend, *, overflow=False):
    full = run_control(directory / "full", accelerator, precision, backend, overflow=overflow)
    prefix = run_control(directory / "prefix", accelerator, precision, backend, epochs=1, overflow=overflow)
    resumed = run_control(
        directory / "resumed", accelerator, precision, backend, checkpoint=prefix.checkpoint, overflow=overflow, lr=0.5
    )
    for audit in (full, prefix, resumed):
        assert_finite(audit.initial)
        assert_finite(audit.final)
        assert_finite(audit.model.scientific_observations)
    expected_steps = 5 if overflow else 6
    assert len(full.batches) == 10 and len(full.preclip) == 6
    assert full.applied_steps == expected_steps
    assert prefix.applied_steps == expected_steps - 3 and resumed.applied_steps == 3
    assert prefix.final["global_step"] == 3 and resumed.initial["global_step"] == 3
    assert resumed.final["global_step"] == 6
    for key in ("weight", "optimizer", "precision_state"):
        assert_same(resumed.initial[key], prefix.final[key])
        assert_same(resumed.final[key], full.final[key])
    assert resumed.initial["optimizer"]["param_groups"][0]["lr"] == 0.125
    assert_same([row["weight"] for row in resumed.batches], [row["weight"] for row in full.batches[5:]])
    assert full.device.split(":")[0] == accelerator and full.precision == precision
    expected_dtype = {"32-true": "torch.float32", "bf16-mixed": "torch.bfloat16", "16-mixed": "torch.float16"}[precision]
    assert {row["prediction_dtype"] for row in full.model.scientific_observations} == {expected_dtype}
    assert {row["input_device"] for row in full.model.scientific_observations} == {accelerator}
    observations = full.model.scientific_observations
    expected_x = [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[2.0, 0.0]], [[0.0, 2.0]]]
    expected_y = [[[0.25]], [[-0.5]], [[0.5]], [[1.0]], [[-1.0]]]
    assert [row["x"] for row in observations] == expected_x * 2
    assert [row["y"] for row in observations] == expected_y * 2
    assert full.loss == pytest.approx(sum(row["loss"] for row in observations[-5:]) / 5, rel=1e-6, abs=1e-7)
    assert torch.isfinite(full.final["weight"]).all()
    assert not torch.equal(full.final["weight"], torch.tensor([[0.5, -0.25]]))
    if precision == "16-mixed":
        assert prefix.final["precision_state"]["scale"] == (16 if overflow else 64)
        assert resumed.final["precision_state"]["scale"] == (128 if overflow else 512)
    else:
        assert full.final["precision_state"] == {}
    if overflow:
        assert_same(full.batches[1]["weight"], full.initial["weight"])
        assert full.batches[1]["optimizer"]["state"] == {}
        assert full.batches[1]["applied_steps"] == 0
        assert torch.isinf(full.preclip[0]["gradient"]).all()
    else:
        assert all(torch.isfinite(row["gradient"]).all() for row in full.preclip)
    return full


@pytest.mark.parametrize(("accelerator", "precision"), PROFILES)
def test_native_accelerator_lifecycle(tmp_path, accelerator, precision):
    require_device(accelerator, precision)
    check_lifecycle(tmp_path / "native", accelerator, precision, "native")


@pytest.mark.parametrize(("accelerator", "precision"), PROFILES)
def test_lighter_accelerator_lifecycle(tmp_path, accelerator, precision):
    require_device(accelerator, precision)
    native = run_control(tmp_path / "native", accelerator, precision, "native")
    lighter = check_lifecycle(tmp_path / "lighter", accelerator, precision, "lighter")
    for key in ("weight", "optimizer", "precision_state"):
        assert_same(lighter.final[key], native.final[key])
    assert_same(lighter.preclip, native.preclip)
    assert_same(
        [row["loss"] for row in lighter.model.scientific_observations],
        [row["loss"] for row in native.model.scientific_observations],
    )
    assert lighter.loss == pytest.approx(native.loss, rel=1e-6, abs=1e-7)


@pytest.mark.parametrize("backend", ["native", "lighter"])
def test_cuda_fp16_nonfinite_update_and_scaler_restore(tmp_path, backend):
    require_device("cuda")
    actual = check_lifecycle(tmp_path / backend, "cuda", "16-mixed", backend, overflow=True)
    if backend == "lighter":
        native = run_control(tmp_path / "native", "cuda", "16-mixed", "native", overflow=True)
        assert_same(actual.preclip, native.preclip)
        for key in ("weight", "optimizer", "precision_state"):
            assert_same(actual.final[key], native.final[key])
