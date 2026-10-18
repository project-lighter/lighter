---
title: Coming from LightningCLI or Hydra
---

# Coming from LightningCLI or Hydra

Lighter does what LightningCLI and Hydra's `instantiate` do for a Lightning project: build the trainer, model
and data from YAML, and change values from the command line. Your LightningModule and LightningDataModule stay
as they are. What changes is how the config is written and run, and [a few things](#what-lighter-does-not-do)
that Lighter leaves to other tools.

## From LightningCLI

| LightningCLI | Lighter |
|--------------|---------|
| A `main.py` that calls `LightningCLI(Model, DataModule)` | No script: `lighter fit config.yaml` |
| `class_path: my.Model` and `init_args:` | `_target_: project.model.Model` and the arguments next to it |
| `--model.lr 0.01` or `--model.lr=0.01` | `model::lr=0.01` |
| `--config a.yaml --config b.yaml` | `lighter fit a.yaml b.yaml` |
| `--print_config` | `--print_config` |
| `--ckpt_path` | `--ckpt_path` |
| `link_arguments` | `"@key"` references |
| `--optimizer Adam --optimizer.lr 0.01` | `optimizer: {_target_: torch.optim.Adam, _mode_: callable, lr: 0.01}` with `LighterModule`, or your own `configure_optimizers` |
| `SaveConfigCallback` writes `config.yaml` to the log folder | Lighter writes `config.yaml` to the log folder and into every checkpoint |
| Arguments checked against type hints before the run | Arguments checked when each object is built |

## From Hydra

| Hydra | Lighter |
|-------|---------|
| `_target_` | `_target_` |
| `_partial_: true` | `_mode_: callable` |
| `${model.lr}` | `"@model::lr"`, or `"%model::lr"` to copy a definition and build it again |
| `${oc.env:DATA}` | `"$os.environ['DATA']"` with `_imports_: {os: os}` |
| `python train.py model.lr=0.01` | `lighter fit config.yaml model::lr=0.01` |
| `+key=value` to add a key | `key=value` (new keys are accepted) |
| `~key` | `~key` |
| `hydra.utils.instantiate(cfg.model)` | `lighter.build("config.yaml")` returns the model, trainer and data |
| `outputs/<date>/<time>/.hydra/config.yaml` | `lightning_logs/version_N/config.yaml` |

## What Lighter does not do

- **Sweeps and launchers.** There is no multirun, sweeper or job launcher. Loop in the shell, or let Optuna or a
  W&B sweep run Lighter ([Recipes](recipes.md#sweep-hyperparameters)).
- **Config groups.** There is no `defaults:` list and no `model=resnet` choice. Put each alternative in its own
  file and stack it on the base config: `lighter fit base.yaml model/resnet.yaml`.
- **Run tracking.** Lighter keeps no record of runs beyond Lightning's: with the default logger each run is a
  `lightning_logs/version_N/` folder with its `config.yaml`, and a logger such as TensorBoard, W&B or MLflow
  compares runs.
