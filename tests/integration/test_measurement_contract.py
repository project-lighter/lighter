"""Real-Trainer contracts for automatic measurements, with native controls."""

from functools import wraps

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping, GradientAccumulationScheduler
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, MetricCollection

from lighter.model import LighterModule


class MeasuredModule(LighterModule):
    """A sample mean whose expected value does not depend on model training."""

    def __init__(self, collection=False):
        network = nn.Linear(1, 1)
        super().__init__(
            network=network,
            optimizer=torch.optim.SGD(network.parameters(), lr=0.1),
            val_metrics=MetricCollection({"mean": MeanMetric()}) if collection else MeanMetric(),
            test_metrics=MetricCollection({"mean": MeanMetric()}) if collection else MeanMetric(),
        )

    def training_step(self, batch, batch_idx):
        return self(batch.reshape(-1, 1)).sum() * 0 + 2

    def validation_step(self, batch, batch_idx):
        self.val_metrics(batch)
        return {"loss": batch.mean()}

    def test_step(self, batch, batch_idx):
        self.test_metrics(batch)
        return {"loss": batch.mean()}


class NativeMeasuredModule(pl.LightningModule):
    """Native self.log proves that logger=False supports callback metrics/reset."""

    def __init__(self):
        super().__init__()
        self.mean = MeanMetric()

    def validation_step(self, batch, batch_idx):
        self.mean(batch)
        self.log("mean", self.mean, on_step=False, on_epoch=True, logger=False)


def _loader(values):
    return DataLoader(torch.tensor(values, dtype=torch.float32), batch_size=2)


def _trainer(tmp_path, external_logger=False, **kwargs):
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=CSVLogger(tmp_path, name="measurements") if external_logger else False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=kwargs.pop("num_sanity_val_steps", 0),
        log_every_n_steps=1,
        **kwargs,
    )


def test_native_measurements_without_logger_reset_between_evaluations(tmp_path):
    model = NativeMeasuredModule()
    trainer = _trainer(tmp_path)
    for values, expected in [([1, 3], 2), ([7, 9], 8)]:
        trainer.validate(model, _loader(values), verbose=False)
        assert trainer.callback_metrics["mean"].item() == pytest.approx(expected)
        assert model.mean.update_count == 0


@pytest.mark.parametrize("external_logger", [False, True])
@pytest.mark.parametrize("collection", [False, True])
@pytest.mark.parametrize("stage,prefix,attribute", [("validate", "val", "val_metrics"), ("test", "test", "test_metrics")])
def test_automatic_measurements_reset_between_evaluations(tmp_path, external_logger, collection, stage, prefix, attribute):
    model = MeasuredModule(collection)
    metric_name = "mean" if collection else "MeanMetric"
    trainer = _trainer(tmp_path, external_logger)
    for values, expected in [([1, 3], 2), ([7, 9], 8)]:
        result = getattr(trainer, stage)(model, _loader(values), verbose=False)
        if external_logger:
            assert result[0][f"{prefix}/metrics/{metric_name}/epoch"] == pytest.approx(expected)
        assert trainer.callback_metrics[f"{prefix}/metrics/{metric_name}/epoch"].item() == pytest.approx(expected)
        assert trainer.callback_metrics[f"{prefix}/loss/epoch"].item() == pytest.approx(expected)
        metric = getattr(model, attribute)
        assert (metric["mean"] if collection else metric).update_count == 0


@pytest.mark.parametrize("external_logger", [False, True])
@pytest.mark.parametrize("sanity_steps", [0, 1])
def test_automatic_metric_can_monitor_early_stopping_without_logger(tmp_path, external_logger, sanity_steps):
    monitor = EarlyStopping(monitor="val/metrics/MeanMetric/epoch", mode="min", patience=0)
    trainer = _trainer(tmp_path, external_logger, max_epochs=5, callbacks=[monitor], num_sanity_val_steps=sanity_steps)
    trainer.fit(MeasuredModule(), _loader([1, 3]), _loader([1, 3, 7, 9]))
    assert trainer.should_stop
    assert monitor.stopped_epoch == 1
    assert monitor.best_score.item() == pytest.approx(5)
    assert trainer.global_step == 2
    assert trainer.callback_metrics["train/loss/epoch"].item() == pytest.approx(2)


class EpochLosses(Callback):
    def __init__(self):
        self.values = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.values.append(trainer.callback_metrics["train/loss/epoch"].item())


class ScientificStep:
    """Same arithmetic and manual optimization in the Lighter/native controls."""

    def _science(self, batch, batch_idx):
        if batch_idx in self.skip_batches:
            return None
        if self.loss_kind == "constant":
            loss = self.network.weight.sum() * 0 + 4
        elif self.loss_kind == "values":
            loss = self.network.weight.sum() * 0 + batch.mean()
        else:
            loss = (self.network(batch.reshape(-1, 1)) - 1).square().mean()
        if not self.automatic_optimization:
            optimizer = self.optimizers()
            if batch_idx % self.manual_accumulation == 0:
                optimizer.zero_grad()
            self.manual_backward(loss / self.manual_accumulation)
            if (batch_idx + 1) % self.manual_accumulation == 0 or batch_idx + 1 == self.trainer.num_training_batches:
                optimizer.step()
        return {"loss": loss} if self.dictionary else loss

    def _init_science(self, manual, factor, dictionary, loss_kind, skip_batches):
        self.automatic_optimization = not manual
        self.manual_accumulation = factor
        self.dictionary = dictionary
        self.loss_kind = loss_kind
        self.skip_batches = skip_batches
        nn.init.constant_(self.network.weight, 0.25)


class ScientificLighterModule(ScientificStep, LighterModule):
    def __init__(self, manual=False, factor=1, dictionary=True, loss_kind="constant", skip_batches=()):
        network = nn.Linear(1, 1, bias=False)
        super().__init__(network=network, optimizer=torch.optim.SGD(network.parameters(), lr=0.001))
        self._init_science(manual, factor, dictionary, loss_kind, skip_batches)

    def training_step(self, batch, batch_idx):
        return self._science(batch, batch_idx)


class ScientificNativeModule(ScientificStep, pl.LightningModule):
    def __init__(self, manual=False, factor=1, dictionary=True, loss_kind="constant", skip_batches=()):
        super().__init__()
        self.network = nn.Linear(1, 1, bias=False)
        self._init_science(manual, factor, dictionary, loss_kind, skip_batches)

    def training_step(self, batch, batch_idx):
        output = self._science(batch, batch_idx)
        if output is not None:
            loss = output["loss"] if isinstance(output, dict) else output
            self.log("train/loss/epoch", loss, on_step=False, on_epoch=True, logger=False)
        return output

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("factor", [1, 2, 4])
@pytest.mark.parametrize("dictionary", [False, True])
def test_scientific_loss_is_independent_of_accumulation(tmp_path, manual, factor, dictionary):
    # Five batches deliberately leave partial accumulation windows for factors 2/4.
    for model_type in [ScientificNativeModule, ScientificLighterModule]:
        model = model_type(manual=manual, factor=factor, dictionary=dictionary)
        recorder = EpochLosses()
        trainer = _trainer(tmp_path, max_epochs=1, callbacks=[recorder], accumulate_grad_batches=1 if manual else factor)
        trainer.fit(model, _loader(range(1, 11)))
        assert recorder.values == pytest.approx([4])
        assert trainer.callback_metrics["train/loss/epoch"].item() == pytest.approx(4)
        if isinstance(model, LighterModule):
            assert trainer.callback_metrics["train/loss/step"].item() == pytest.approx(4)


def test_scientific_loss_with_changing_accumulation(tmp_path):
    for model_type in [ScientificNativeModule, ScientificLighterModule]:
        recorder = EpochLosses()
        trainer = _trainer(
            tmp_path,
            max_epochs=3,
            callbacks=[GradientAccumulationScheduler(scheduling={0: 1, 1: 2, 2: 4}), recorder],
        )
        trainer.fit(model_type(), _loader(range(1, 11)))
        assert recorder.values == pytest.approx([4, 4, 4])


@pytest.mark.parametrize("manual", [False, True])
def test_skipped_steps_do_not_reuse_previous_loss(tmp_path, manual):
    for model_type in [ScientificNativeModule, ScientificLighterModule]:
        model = model_type(manual=manual, loss_kind="values", skip_batches=(1, 3, 4))
        recorder = EpochLosses()
        trainer = _trainer(tmp_path, max_epochs=1, callbacks=[recorder])
        trainer.fit(model, _loader([2, 2, 50, 50, 6, 6, 50, 50, 50, 50]))
        assert recorder.values == pytest.approx([4])


@pytest.mark.parametrize("manual", [False, True])
def test_loss_observation_preserves_native_parameter_updates(tmp_path, manual):
    parameters = []
    for model_type in [ScientificNativeModule, ScientificLighterModule]:
        model = model_type(manual=manual, factor=2, loss_kind="quadratic")
        trainer = _trainer(tmp_path, max_epochs=1, accumulate_grad_batches=1 if manual else 2)
        trainer.fit(model, _loader(range(1, 11)))
        parameters.append(model.network.weight.detach().clone())
    assert not torch.equal(parameters[0], torch.full_like(parameters[0], 0.25))
    torch.testing.assert_close(parameters[1], parameters[0], rtol=0, atol=0)


class OuterSkippedModule(ScientificLighterModule):
    def training_step(self, batch, batch_idx):
        output = super().training_step(batch, batch_idx)
        return None if batch_idx == 1 else output


def test_outer_training_step_owns_the_observed_loss(tmp_path):
    recorder = EpochLosses()
    trainer = _trainer(tmp_path, max_epochs=1, callbacks=[recorder])
    trainer.fit(OuterSkippedModule(loss_kind="values"), _loader([2, 2, 50, 50, 6, 6]))
    assert recorder.values == pytest.approx([4])


class SharedScientificStep:
    def training_step(self, batch, batch_idx):
        return self._science(batch, batch_idx)


class MixinLossModule(SharedScientificStep, ScientificLighterModule):
    pass


class InheritedMixinLossModule(MixinLossModule):
    pass


class DecoratedLossModule(ScientificLighterModule):
    @wraps(ScientificLighterModule.training_step)
    def training_step(self, batch, batch_idx):
        output = super().training_step(batch, batch_idx)
        return {"loss": output["loss"] + 3}


@pytest.mark.parametrize("model_type,expected", [(InheritedMixinLossModule, 4), (DecoratedLossModule, 7)])
def test_effective_inherited_or_decorated_step_owns_loss(tmp_path, model_type, expected):
    recorder = EpochLosses()
    trainer = _trainer(tmp_path, max_epochs=1, callbacks=[recorder], accumulate_grad_batches=4)
    trainer.fit(model_type(), _loader(range(1, 11)))
    assert recorder.values == pytest.approx([expected])


class PopulationModule(MeasuredModule):
    def __init__(self, collection=False, persistent=False):
        super().__init__(collection=collection)
        if collection:
            self.val_metrics = MetricCollection({"mean": MeanMetric()}, prefix="sample_", postfix="_value")
            self.test_metrics = MetricCollection({"mean": MeanMetric()}, prefix="sample_", postfix="_value")
        if persistent:
            self.val_metrics.persistent(True)
            self.test_metrics.persistent(True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.val_metrics(batch)
        return batch.mean()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.test_metrics(batch)
        return batch.mean()


class NativePopulationModule(pl.LightningModule):
    def __init__(self, collection=False):
        super().__init__()
        self.population_metrics = nn.ModuleList(
            [
                MetricCollection({"mean": MeanMetric()}, prefix="sample_", postfix="_value") if collection else MeanMetric()
                for _ in range(3)
            ]
        )

    def _measure(self, batch, dataloader_idx, prefix):
        metrics = self.population_metrics[dataloader_idx]
        metrics(batch)
        if isinstance(metrics, MetricCollection):
            metric = metrics["mean"]
            name = "sample_mean_value"
            attribute = f"population_metrics.{dataloader_idx}.mean"
        else:
            metric = metrics
            name = "MeanMetric"
            attribute = f"population_metrics.{dataloader_idx}"
        self.log(
            f"{prefix}/metrics/{name}/epoch",
            metric,
            on_step=False,
            on_epoch=True,
            logger=self.trainer.logger is not None,
            metric_attribute=attribute,
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._measure(batch, dataloader_idx, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self._measure(batch, dataloader_idx, "test")


@pytest.mark.parametrize("stage,prefix", [("validate", "val"), ("test", "test")])
@pytest.mark.parametrize("collection", [False, True])
@pytest.mark.parametrize("external_logger", [False, True])
def test_evaluation_metrics_keep_population_ownership(tmp_path, stage, prefix, collection, external_logger):
    sequences = [
        ([[0, 0, 0], [10]], [0, 10]),
        ([[10], [0, 0, 0]], [10, 0]),
        ([[7, 7]], [7]),
        ([[3], [9, 9, 9], [15]], [3, 9, 15]),
    ]
    name = "sample_mean_value" if collection else "MeanMetric"
    for model_type in [NativePopulationModule, PopulationModule]:
        model = model_type(collection=collection)
        trainer = _trainer(tmp_path, external_logger)
        for populations, expected in sequences:
            getattr(trainer, stage)(model, [_loader(values) for values in populations], verbose=False)
            for index, mean in enumerate(expected):
                suffix = f"/dataloader_idx_{index}" if len(populations) > 1 else ""
                assert trainer.callback_metrics[f"{prefix}/metrics/{name}/epoch{suffix}"].item() == pytest.approx(mean)


class PopulationValidationHistory(Callback):
    def __init__(self):
        self.means = []

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self.means.append(
            [trainer.callback_metrics[f"val/metrics/MeanMetric/epoch/dataloader_idx_{index}"].item() for index in range(2)]
        )


def test_population_metrics_survive_sanity_fit_validate_and_test(tmp_path):
    model = PopulationModule()
    original_val, original_test = model.val_metrics, model.test_metrics
    history = PopulationValidationHistory()
    trainer = _trainer(tmp_path, max_epochs=2, num_sanity_val_steps=1, callbacks=[history])
    populations = [_loader([0, 0, 6]), _loader([10])]
    trainer.fit(model, _loader([1, 3]), populations)
    assert len(history.means) == 2
    for observed, expected in zip(history.means, [[2, 10], [2, 10]], strict=True):
        assert observed == pytest.approx(expected)
    trainer.validate(model, [_loader([8]), _loader([4, 4, 4])], verbose=False)
    assert history.means[-1] == pytest.approx([8, 4])
    trainer.test(model, [_loader([3]), _loader([9, 9, 9])], verbose=False)
    assert trainer.callback_metrics["test/metrics/MeanMetric/epoch/dataloader_idx_0"].item() == pytest.approx(3)
    assert trainer.callback_metrics["test/metrics/MeanMetric/epoch/dataloader_idx_1"].item() == pytest.approx(9)
    assert model.val_metrics is original_val
    assert model.test_metrics is original_test
    assert all(metric.update_count == 0 for metric in model.modules() if isinstance(metric, MeanMetric))


def test_persistent_metric_single_loader_state_round_trip():
    model = PopulationModule(persistent=True)
    model.validation_step(torch.tensor([2.0, 4.0]), 0)
    state = model.state_dict()
    assert "val_metrics.mean_value" in state
    restored = PopulationModule(persistent=True)
    restored.load_state_dict(state)
    assert restored.val_metrics.compute().item() == pytest.approx(3)


@pytest.mark.parametrize("collection", [False, True])
def test_automatic_population_cloning_rejects_checkpoint_state(tmp_path, collection):
    trainer = _trainer(tmp_path)
    with pytest.raises(ValueError, match="checkpoint state.*per.dataloader"):
        trainer.validate(
            PopulationModule(collection=collection, persistent=True), [_loader([0, 0]), _loader([10])], verbose=False
        )


class TensorOnlyEvaluation(LighterModule):
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return batch.mean()


def test_multiple_loaders_without_metrics_keep_native_loss_logging(tmp_path):
    model = TensorOnlyEvaluation(network=nn.Identity())
    trainer = _trainer(tmp_path)
    trainer.validate(model, [_loader([0, 0, 0]), _loader([10])], verbose=False)
    for index, expected in enumerate([0, 10]):
        assert trainer.callback_metrics[f"val/loss/epoch/dataloader_idx_{index}"].item() == pytest.approx(expected)


class DelegatedPopulationModule(PopulationModule):
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # The outer native invocation owns the population, even when the parent
        # method does not need the index for its scientific calculation.
        return super().validation_step(batch, batch_idx)


def test_outer_evaluation_step_owns_population_during_super_call(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.validate(DelegatedPopulationModule(), [_loader([0, 0, 0]), _loader([10])], verbose=False)
    for index, expected in enumerate([0, 10]):
        key = f"val/metrics/MeanMetric/epoch/dataloader_idx_{index}"
        assert trainer.callback_metrics[key].item() == pytest.approx(expected)


class CloneBeforeUpdateMean(MeanMetric):
    def clone(self):
        # Metrics such as AUROC may retain large prediction buffers. Cloning
        # those after loader0 runs would copy the full population before reset.
        assert self.update_count == 0, "clone must precede population updates"
        return super().clone()


def test_loader_metrics_are_cloned_before_population_data_accumulates(tmp_path):
    model = PopulationModule()
    model.val_metrics = CloneBeforeUpdateMean()
    trainer = _trainer(tmp_path)
    trainer.validate(model, [_loader([0, 0, 0]), _loader([10])], verbose=False)
    for index, expected in enumerate([0, 10]):
        key = f"val/metrics/CloneBeforeUpdateMean/epoch/dataloader_idx_{index}"
        assert trainer.callback_metrics[key].item() == pytest.approx(expected)
