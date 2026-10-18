---
title: LighterModule
---

# LighterModule

Any LightningModule works with Lighter. `LighterModule` is an optional base class for the common case: it takes
the network, loss, optimizer, scheduler and metrics from the config, and logs the loss and metrics for you.
You still write every step method, as in Lightning.

This is [`projects/tabular_regression`](https://github.com/project-lighter/lighter/tree/main/projects/tabular_regression),
which a test runs:

```python title="task.py"
--8<-- "projects/tabular_regression/task.py"
```

```yaml title="config.yaml"
--8<-- "projects/tabular_regression/config.yaml"
```

## What it does

`LighterModule.__init__` takes `network`, `criterion`, `optimizer`, `scheduler`, `train_metrics`, `val_metrics`
and `test_metrics`, and stores them as attributes of the same names. Only `network` is required. On top of
Lightning it adds three things:

- `forward` calls `self.network`.
- `configure_optimizers` returns the optimizer and the scheduler from the config (below).
- After each train, validation and test step it logs what the step returned (below).

There are no default steps: without a `validation_step`, Lightning skips validation, as with any
LightningModule. To add arguments of your own, write an `__init__` and pass LighterModule's arguments on to
`super().__init__`. Do not call `self.save_hyperparameters()`: it would store the optimizer function in every
checkpoint, which `lighter.load_model` then cannot load safely. Lighter already saves the config with the run.

## Optimizer and scheduler

With `_mode_: callable`, `optimizer` is a function that LighterModule calls with `self.parameters()`, and
`scheduler` one that it calls with the optimizer. When the scheduler needs Lightning's settings, such as
`monitor` for `ReduceLROnPlateau`, give a dict with Lightning's keys:

```yaml
scheduler:
  scheduler:
    _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
    _mode_: callable
    patience: 3
  monitor: val/loss/epoch
  interval: epoch
```

An optimizer that needs parts of the network, such as parameter groups with their own learning rates, can be
built in the config instead. Then build the scheduler there too, with `optimizer: "@model::optimizer"`.

## What gets logged

A step may return the loss tensor, a dict with a `"loss"` key (the other keys are passed on to callbacks), or
None; anything else, such as a tuple, logs no loss and warns. For each stage (`train`, `val`, `test`) LighterModule logs:

| Logged as | Value |
|-----------|-------|
| `<stage>/loss/step`, `<stage>/loss/epoch` | the loss the step returned |
| `<stage>/metrics/<name>/step`, `<stage>/metrics/<name>/epoch` | each `<stage>_metrics` that the step updated |

`<name>` is the metric's class name (`torchmetrics.Accuracy` with `task: multiclass` is `MulticlassAccuracy`), or
its key when `metrics:` of a `MetricCollection` is a dict. Values are logged also when `trainer::logger` is
false, so `ModelCheckpoint` and `EarlyStopping` can monitor them (e.g. `monitor: val/loss/epoch`).

The loss must be one tensor, the one that is backpropagated. Log its parts yourself with
`self.log("train/loss/ce", ce)`. The learning rate is logged by Lightning's `LearningRateMonitor` callback.

Each `*_metrics` is one torchmetrics `Metric` or a `MetricCollection`. For several metrics, use a
`MetricCollection`; for validation and test, copy the definition with `%` so each stage counts on its own:

```yaml
train_metrics:
  _target_: torchmetrics.MetricCollection
  metrics:
    - _target_: torchmetrics.Accuracy
      task: multiclass
      num_classes: 10
    - _target_: torchmetrics.F1Score
      task: multiclass
      num_classes: 10
val_metrics: "%train_metrics"
```

One metric object cannot serve several validation or test dataloaders; with more than one, create a metric per
dataloader in your module and log each with `self.log`.

## LighterDataModule

`lighter.LighterDataModule` holds the four dataloaders from the config: `train_dataloader`, `val_dataloader`,
`test_dataloader` and `predict_dataloader`. Leave out the ones you do not use; a stage whose dataloader is
missing (e.g. `lighter test` without `test_dataloader`) names the key to add. When your data needs
`prepare_data` or a `setup` that depends on the stage, write a LightningDataModule and point `data::_target_`
at it.
