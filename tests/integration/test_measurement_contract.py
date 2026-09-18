"""Real-Trainer contracts for automatic measurements, with native controls."""

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
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
