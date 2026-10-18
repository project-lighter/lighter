"""Tests for LighterModule: optimizer building, step outputs and logging, run through a real Trainer."""

from functools import partial

import pytest
import torch
from pytorch_lightning import Trainer
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, MetricCollection

from lighter import LighterDataModule, LighterModule
from lighter.engine.runner import load_config


def loader(size=8):
    return DataLoader(TensorDataset(torch.randn(size, 4), torch.randint(0, 2, (size,))), batch_size=4)


def trainer(**kwargs):
    options = dict(accelerator="cpu", max_epochs=1, logger=False, enable_checkpointing=False, enable_progress_bar=False)
    return Trainer(**options, enable_model_summary=False, num_sanity_val_steps=0, **kwargs)


class Task(LighterModule):
    """Returns `{"loss", "pred"}` from training and `val_output(loss)` from validation."""

    val_output = staticmethod(lambda loss: loss)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        if self.train_metrics is not None:
            self.train_metrics(pred, y)
        return {"loss": torch.nn.functional.cross_entropy(pred, y), "pred": pred}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        pred = self(x)
        if self.val_metrics is not None:
            self.val_metrics(pred, y)
        return self.val_output(torch.nn.functional.cross_entropy(pred, y))


def task(**kwargs):
    kwargs.setdefault("optimizer", partial(SGD, lr=0.1))
    return Task(network=torch.nn.Linear(4, 2), **kwargs)


def test_forward_calls_the_network():
    model = task()
    x = torch.randn(3, 4)
    assert torch.equal(model(x), model.network(x))


@pytest.mark.parametrize("metrics", [[Accuracy(task="binary")], {"acc": Accuracy(task="binary")}])
def test_metrics_must_be_metric_objects(metrics):
    with pytest.raises(TypeError, match="`val_metrics` must be a torchmetrics Metric or MetricCollection, got"):
        task(val_metrics=metrics)


# Optimizer and scheduler


def test_prebuilt_optimizer_and_scheduler_are_used_as_given():
    model = task(optimizer=None)
    model.optimizer = SGD(model.parameters(), lr=0.1)
    assert model.configure_optimizers() is model.optimizer
    model.scheduler = StepLR(model.optimizer, step_size=1)
    assert model.configure_optimizers() == {"optimizer": model.optimizer, "lr_scheduler": {"scheduler": model.scheduler}}


def test_callable_optimizer_and_scheduler_are_built_from_the_model():
    model = task(scheduler={"scheduler": partial(ReduceLROnPlateau), "monitor": "val/loss/epoch"})
    config = model.configure_optimizers()
    assert config["optimizer"].param_groups[0]["params"] == list(model.parameters())
    assert isinstance(config["lr_scheduler"]["scheduler"], ReduceLROnPlateau)
    assert config["lr_scheduler"]["scheduler"].optimizer is config["optimizer"]
    assert config["lr_scheduler"]["monitor"] == "val/loss/epoch"
    assert callable(model.scheduler["scheduler"])  # the config is not consumed, so it can be built again


def test_callable_optimizer_with_a_prebuilt_scheduler_is_an_error():
    other = SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1)
    with pytest.raises(ValueError, match="give the scheduler `_mode_: callable` too"):
        task(scheduler=StepLR(other, step_size=1)).configure_optimizers()


def test_missing_optimizer_is_an_error():
    with pytest.raises(ValueError, match="set `model::optimizer`"):
        task(optimizer=None).configure_optimizers()


def test_optimizer_and_scheduler_from_yaml_with_mode_callable(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "model:\n"
        "  _target_: lighter.LighterModule\n"
        "  network: {_target_: torch.nn.Linear, in_features: 4, out_features: 2}\n"
        "  optimizer: {_target_: torch.optim.SGD, _mode_: callable, lr: 0.1}\n"
        "  scheduler:\n"
        "    scheduler: {_target_: torch.optim.lr_scheduler.ReduceLROnPlateau, _mode_: callable, patience: 2}\n"
        "    monitor: val/loss/epoch\n"
    )
    model = load_config(str(tmp_path / "config.yaml")).resolve("model")
    config = model.configure_optimizers()
    assert config["optimizer"].param_groups[0]["params"] == list(model.parameters())
    assert config["lr_scheduler"]["scheduler"].patience == 2


# Logging


def test_train_loss_is_logged_as_returned_under_gradient_accumulation():
    class ConstantLoss(Task):
        def training_step(self, batch, batch_idx):
            return self.network.weight.sum() * 0 + 8.0

    run = trainer(accumulate_grad_batches=2)
    run.fit(ConstantLoss(network=torch.nn.Linear(4, 2), optimizer=partial(SGD, lr=0.1)), loader(16))
    assert run.callback_metrics["train/loss/epoch"] == 8.0


def test_losses_and_metrics_are_logged_without_a_logger():
    run = trainer()
    metrics = MetricCollection([Accuracy(task="multiclass", num_classes=2)])
    run.fit(task(train_metrics=metrics.clone(), val_metrics=metrics.clone()), loader(), loader())
    names = ["train/loss/step", "train/metrics/MulticlassAccuracy/step"]
    names += [f"{mode}/{name}/epoch" for mode in ("train", "val") for name in ("loss", "metrics/MulticlassAccuracy")]
    assert set(names) <= set(run.callback_metrics)
    assert {"train/loss/step", "val/loss/epoch"} <= set(run.progress_bar_metrics)


@pytest.mark.parametrize("output", [None, {"pred": 1}])
def test_validation_step_may_return_no_loss(output, recwarn):
    run = trainer()
    model = task(val_metrics=Accuracy(task="multiclass", num_classes=2))
    model.val_output = lambda loss: output
    run.fit(model, loader(), loader())
    assert "val/metrics/MulticlassAccuracy/epoch" in run.callback_metrics
    assert "val/loss/epoch" not in run.callback_metrics
    assert not [warning for warning in recwarn if "no loss is logged" in str(warning.message)]


def test_a_step_output_without_a_loss_warns():
    model = task()
    model.val_output = lambda loss: (loss, loss)
    with pytest.warns(UserWarning, match="The val step returned tuple, so no loss is logged"):
        trainer().validate(model, loader())


def test_a_float_loss_is_logged():
    run = trainer()
    model = task()
    model.val_output = lambda loss: loss.item()
    run.validate(model, loader())
    assert "val/loss/epoch" in run.callback_metrics


def test_a_dict_under_loss_explains_the_migration():
    model = task()
    model.val_output = lambda loss: {"loss": {"total": loss, "ce": loss}}
    with pytest.raises(TypeError, match=r"Return \{'loss': total\} and log the parts yourself"):
        trainer().validate(model, loader())


def test_one_metric_for_several_dataloaders_is_an_error():
    model = task(val_metrics=Accuracy(task="multiclass", num_classes=2))
    with pytest.raises(ValueError, match="`val_metrics` would mix the results of several dataloaders"):
        trainer().validate(model, [loader(), loader()])


# Stages with missing steps or dataloaders


def test_fit_with_only_a_training_step_and_train_dataloader():
    class TrainOnly(LighterModule):
        def training_step(self, batch, batch_idx):
            return torch.nn.functional.cross_entropy(self(batch[0]), batch[1])

    run = trainer()
    run.fit(TrainOnly(network=torch.nn.Linear(4, 2), optimizer=partial(SGD, lr=0.1)), datamodule=LighterDataModule(loader()))
    assert run.global_step == 2


def test_val_dataloader_without_validation_step_is_skipped():
    class TrainOnly(Task):
        validation_step = LighterModule.validation_step

    run = trainer()
    run.fit(TrainOnly(network=torch.nn.Linear(4, 2), optimizer=partial(SGD, lr=0.1)), loader(), loader())
    assert "val/loss/epoch" not in run.callback_metrics and run.global_step == 2


def test_a_stage_without_its_dataloader_names_the_key_to_add():
    from lighter.engine.runner import _check_stage_data

    datamodule = LighterDataModule(train_dataloader=loader())
    _check_stage_data("fit", datamodule)
    with pytest.raises(ValueError, match=r"`lighter validate` needs `data::val_dataloader`, which is not set"):
        _check_stage_data("validate", datamodule)


def test_train_only_datamodule_fits_without_dataloader_warnings(recwarn):
    class TrainOnly(Task):
        validation_step = LighterModule.validation_step

    model = TrainOnly(network=torch.nn.Linear(4, 2), optimizer=partial(SGD, lr=0.1))
    trainer().fit(model, datamodule=LighterDataModule(train_dataloader=loader()))
    messages = [str(warning.message) for warning in recwarn]
    assert not [message for message in messages if "val_dataloader" in message or "across ranks is zero" in message]
