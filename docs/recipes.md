---
title: Recipes
---

# Recipes

Things you do once a project runs, shown on
[`projects/tabular_regression`](https://github.com/project-lighter/lighter/tree/main/projects/tabular_regression).
The files are in its `recipes/` folder. A test copies the project and runs every `lighter` command on this page,
in order, from inside that folder.

## Try a variant

Config files merge from left to right, so a variant is a small file stacked on the base config:

```yaml title="recipes/mlp.yaml"
--8<-- "projects/tabular_regression/recipes/mlp.yaml"
```

```bash
lighter fit config.yaml recipes/mlp.yaml
```

Lighter has no config groups. Keep alternatives in files, such as `model/mlp.yaml` and `model/resnet.yaml`, and
name the ones a run uses on the command line.

## Resume training

```yaml title="recipes/resume.yaml"
--8<-- "projects/tabular_regression/recipes/resume.yaml"
```

```bash
lighter fit config.yaml recipes/resume.yaml
lighter fit config.yaml recipes/resume.yaml --ckpt_path last trainer::max_epochs=30
```

The second run loads `checkpoints/last.ckpt`, with the optimizer and scheduler state, and trains epochs 20 to 29
in a new version folder. Without a fixed `dirpath`, each run checkpoints into its own version folder, so
`--ckpt_path last` finds nothing and stops with an error instead of training from scratch; pass the checkpoint
file instead.

## Repeat a run

Each run saves its composed config, seed included, in its log folder. Running that config repeats the run:

```bash
lighter fit lightning_logs/version_0/config.yaml
```

## Seed a run

The seed is the top-level `seed:` key, 0 unless you set it. Lighter passes it
to Lightning's `seed_everything(seed, workers=True)`, which seeds Python, NumPy, torch and the DataLoader
workers. Set it in the config (`seed: 42`) or for one run:

```bash
lighter fit config.yaml seed=7
```

For results that are identical on a GPU too, also set `trainer::deterministic=true`.

## Freeze layers

`lighter.callbacks.Freezer` freezes parameters by name or name prefix, for the whole run or until a step or
epoch (`until_step`, `until_epoch`). With a LighterModule, names are relative to `network`:

```yaml title="recipes/freeze.yaml"
--8<-- "projects/tabular_regression/recipes/freeze.yaml"
```

```bash
lighter fit config.yaml recipes/mlp.yaml recipes/freeze.yaml
```

## Write predictions to a file

```yaml title="recipes/predictions.yaml"
--8<-- "projects/tabular_regression/recipes/predictions.yaml"
```

```bash
lighter predict config.yaml recipes/predictions.yaml --ckpt_path checkpoints/last.ckpt --no-return_predictions
```

`--no-return_predictions` stops Lightning from also keeping every prediction in memory. For tensors or images,
[`FileWriter`][lighter.callbacks.file_writer.FileWriter] writes one file per sample instead.

## Sweep hyperparameters

Lighter does not run sweeps. Loop in the shell:

```bash
for lr in 0.3 0.1 0.03; do lighter fit config.yaml model::optimizer::lr=$lr; done
```

Or let a sweep tool choose the values. With Optuna, build each trial's run with `lighter.build`, which takes the
same inputs as the command line, and read the result from the trainer:

```python title="recipes/optuna_sweep.py"
--8<-- "projects/tabular_regression/recipes/optuna_sweep.py"
```

A W&B sweep runs the command you give it, with the sampled values as overrides:

```yaml title="recipes/wandb_sweep.yaml"
--8<-- "projects/tabular_regression/recipes/wandb_sweep.yaml"
```

`wandb sweep recipes/wandb_sweep.yaml` creates the sweep and prints the `wandb agent` command that runs it.

## Log to another service

`trainer::logger` takes any Lightning logger, and Lighter logs the composed config as the run's
hyperparameters. Without one, Lightning uses TensorBoard when it is installed, else a CSV file.

```yaml title="recipes/wandb.yaml"
--8<-- "projects/tabular_regression/recipes/wandb.yaml"
```

```bash
lighter fit config.yaml recipes/wandb.yaml
```

## Train on several GPUs

```bash
lighter fit config.yaml trainer::accelerator=gpu trainer::devices=4 trainer::strategy=ddp
```

Lightning starts one process per device. Each builds the run from the same config, and only the first prints the
banner and saves the config. `batch_size` is per process. A test runs this with two CPU processes
(`trainer::accelerator=cpu trainer::devices=2`).

## Change one callback from the command line

A list item has no name, so an override cannot pick one callback out of `trainer::callbacks`. Give them names
in a dict, and turn it into the list the trainer takes:

```yaml title="recipes/callbacks.yaml"
--8<-- "projects/tabular_regression/recipes/callbacks.yaml"
```

```bash
lighter fit config.yaml recipes/callbacks.yaml callbacks::early_stopping::patience=5
lighter fit config.yaml recipes/callbacks.yaml callbacks::early_stopping::_disabled_=true
```

A disabled callback is left out of the list. Another file can change one the same way, by its name.

## Use a trained model in a notebook

```python title="recipes/load_model.py"
--8<-- "projects/tabular_regression/recipes/load_model.py"
```

`lighter.load_model` builds the model from the config saved in the checkpoint, with any overrides you pass after
the path (`lighter.load_model(path, "model::criterion=null")`), and loads the weights. A checkpoint saved with
`save_weights_only=True` holds no config: build the model with
`lighter.build("lightning_logs/version_0/config.yaml")` and load the weights yourself.
