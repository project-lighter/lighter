"""Native Lightning contracts for optional stages and scientific step outputs."""

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

from lighter import LighterDataModule, LighterModule


def loader(values=(1, 3, 8)):
    return DataLoader(torch.tensor(values, dtype=torch.float32), batch_size=2)


def trainer(tmp_path, external_logger=False, **kwargs):
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=CSVLogger(tmp_path) if external_logger else False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        **kwargs,
    )


class NativeNoneEvaluation(pl.LightningModule):
    def validation_step(self, batch, batch_idx):
        self.log("explicit", batch.mean(), on_epoch=True, logger=self.trainer.logger is not None)

    def test_step(self, batch, batch_idx):
        self.log("explicit", batch.mean(), on_epoch=True, logger=self.trainer.logger is not None)


class NoneEvaluation(LighterModule):
    def __init__(self):
        super().__init__(network=nn.Identity(), val_metrics=MeanMetric(), test_metrics=MeanMetric())

    def validation_step(self, batch, batch_idx):
        self.val_metrics(batch)
        self.log("explicit", batch.mean(), on_epoch=True, logger=self.trainer.logger is not None)

    def test_step(self, batch, batch_idx):
        self.test_metrics(batch)
        self.log("explicit", batch.mean(), on_epoch=True, logger=self.trainer.logger is not None)


@pytest.mark.parametrize("stage,prefix", [("validate", "val"), ("test", "test")])
@pytest.mark.parametrize("external_logger", [False, True])
def test_none_evaluation_output_retains_native_and_automatic_logs(tmp_path, stage, prefix, external_logger):
    for model_type in [NativeNoneEvaluation, NoneEvaluation]:
        engine = trainer(tmp_path, external_logger)
        model = model_type()
        getattr(engine, stage)(model, loader(), verbose=False)
        assert engine.callback_metrics["explicit"].item() == pytest.approx(4)
        if isinstance(model, LighterModule):
            assert engine.callback_metrics[f"{prefix}/metrics/MeanMetric/epoch"].item() == pytest.approx(4)


class TrainingOnly(LighterModule):
    def __init__(self):
        network = nn.Linear(1, 1)
        super().__init__(network=network, optimizer=torch.optim.SGD(network.parameters(), lr=0.01))

    def training_step(self, batch, batch_idx):
        return self.network(batch.reshape(-1, 1)).sum() * 0 + 4


class NativeTrainingOnly(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        return self.network(batch.reshape(-1, 1)).sum() * 0 + 4

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class NativeTrainingData(pl.LightningDataModule):
    def train_dataloader(self):
        return loader()


class RejectPhantomValidation(Callback):
    def on_validation_start(self, trainer, pl_module):
        raise AssertionError("training-only run unexpectedly entered validation")


def test_training_only_data_and_model_have_no_phantom_validation(tmp_path, recwarn):
    for model, data in [
        (NativeTrainingOnly(), NativeTrainingData()),
        (TrainingOnly(), LighterDataModule(train_dataloader=loader())),
    ]:
        assert not is_overridden("validation_step", model)
        assert not is_overridden("val_dataloader", data)
        engine = trainer(tmp_path, max_epochs=1, callbacks=[RejectPhantomValidation()])
        engine.fit(model, datamodule=data)
        assert engine.global_step == 2
    assert not any(
        "validation_step" in str(warning.message) or "val_dataloader" in str(warning.message) for warning in recwarn
    )


class CustomData(LighterDataModule):
    def train_dataloader(self):
        return loader()

    def val_dataloader(self):
        return loader()


class PlateauModel(TrainingOnly):
    def __init__(self):
        super().__init__()
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=0),
            "monitor": "quality",
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }

    def validation_step(self, batch, batch_idx):
        self.log("quality", batch.mean(), on_epoch=True, logger=False)


class NativePlateauModel(NativeTrainingOnly):
    def __init__(self):
        super().__init__()
        self.optimizer = super().configure_optimizers()
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=0),
            "monitor": "quality",
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }

    def validation_step(self, batch, batch_idx):
        self.log("quality", batch.mean(), on_epoch=True, logger=False)

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}


def test_custom_loader_hooks_and_native_plateau_scheduler_dictionary(tmp_path):
    for model in [NativePlateauModel(), PlateauModel()]:
        data = CustomData()
        assert is_overridden("train_dataloader", data)
        assert is_overridden("val_dataloader", data)
        assert model.configure_optimizers()["lr_scheduler"] is model.scheduler
        settings = dict(model.scheduler)
        engine = trainer(tmp_path, max_epochs=2)
        engine.fit(model, datamodule=data)
        assert model.optimizer.param_groups[0]["lr"] == pytest.approx(0.005)
        assert all(model.scheduler[key] == value for key, value in settings.items())
        assert model.scheduler["reduce_on_plateau"] is True
        assert engine.callback_metrics["quality"].item() == pytest.approx(4)


class NestedTrainingLoss(TrainingOnly):
    def training_step(self, batch, batch_idx):
        total = super().training_step(batch, batch_idx)
        return {"loss": {"total": total, "part": total}}


def test_nested_training_loss_fails_with_actionable_native_contract(tmp_path):
    with pytest.raises(TypeError, match="loss.*Tensor.*loss_terms"):
        trainer(tmp_path, max_epochs=1).fit(NestedTrainingLoss(), loader())


class NamedTrainingLoss(TrainingOnly):
    def training_step(self, batch, batch_idx):
        total = super().training_step(batch, batch_idx)
        return {"loss": total, "loss_terms": {"first": total / 4, "second": total * 0.75}}


def test_named_loss_terms_keep_scalar_optimization_loss(tmp_path):
    engine = trainer(tmp_path, max_epochs=1, accumulate_grad_batches=2)
    engine.fit(NamedTrainingLoss(), loader())
    assert engine.callback_metrics["train/loss/epoch"].item() == pytest.approx(4)
    assert engine.callback_metrics["train/loss/first/epoch"].item() == pytest.approx(1)
    assert engine.callback_metrics["train/loss/second/epoch"].item() == pytest.approx(3)


class ManualNoLoss(TrainingOnly):
    def __init__(self, return_none):
        super().__init__()
        self.automatic_optimization = False
        self.return_none = return_none

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        prediction = self(batch.reshape(-1, 1))
        self.manual_backward(prediction.square().mean())
        optimizer.step()
        return None if self.return_none else {"prediction": prediction.detach()}


@pytest.mark.parametrize("return_none", [False, True])
def test_manual_optimization_does_not_require_returned_loss(tmp_path, return_none):
    model = ManualNoLoss(return_none)
    initial = model.network.weight.detach().clone()
    engine = trainer(tmp_path, max_epochs=1)
    engine.fit(model, loader())
    assert engine.global_step == 2
    assert not torch.equal(initial, model.network.weight.detach())


@pytest.mark.parametrize("loss", [4.0, torch.tensor([1.0, 2.0])])
def test_automatic_optimization_rejects_non_scalar_tensor_loss_early(loss):
    class InvalidLoss(TrainingOnly):
        def training_step(self, batch, batch_idx):
            return {"loss": loss}

    with pytest.raises(TypeError, match="single.element Tensor"):
        InvalidLoss().training_step(None, 0)


class IntermediateStep(TrainingOnly):
    def training_step(self, batch, batch_idx):
        return {"prediction": self(batch.reshape(-1, 1))}


class OuterPredictionLoss(IntermediateStep):
    def training_step(self, batch, batch_idx):
        output = super().training_step(batch, batch_idx)
        return output["prediction"].square().mean()


class NativeIntermediateStep(NativeTrainingOnly):
    def training_step(self, batch, batch_idx):
        return {"prediction": self.network(batch.reshape(-1, 1))}


class NativeOuterPredictionLoss(NativeIntermediateStep):
    def training_step(self, batch, batch_idx):
        output = super().training_step(batch, batch_idx)
        return output["prediction"].square().mean()


def test_only_outer_training_step_return_must_supply_native_loss(tmp_path):
    native = NativeOuterPredictionLoss()
    model = OuterPredictionLoss()
    model.load_state_dict(native.state_dict())
    for task in (native, model):
        engine = trainer(tmp_path, max_epochs=1)
        engine.fit(task, loader())
        assert engine.global_step == 2
    torch.testing.assert_close(model.network.weight, native.network.weight, rtol=0, atol=0)
    torch.testing.assert_close(model.network.bias, native.network.bias, rtol=0, atol=0)


def test_training_output_error_does_not_disable_later_validation():
    class ReturnedInput(TrainingOnly):
        def training_step(self, batch, batch_idx):
            return batch

    model = ReturnedInput()
    for invalid in (4.0, {"prediction": torch.tensor(4.0)}):
        with pytest.raises(TypeError, match="single.element Tensor"):
            model.training_step(invalid, 0)
        loss = torch.tensor(4.0, requires_grad=True)
        assert model.training_step(loss, 0) is loss
