# Changelog

All notable changes to Lighter are listed here. Breaking changes come with a one-line migration.

## Unreleased

### Added

- A top-level `seed:` key (default: `PL_GLOBAL_SEED`, else 0) seeds the run, including DataLoader workers,
  and is written into the saved config. Migration: rename any top-level `seed` key you used for something else.
- `python -m lighter` runs the CLI.
- `lighter.build(*inputs)` returns `(model, trainer, datamodule)` without running a stage, and `Runner.run`
  returns the stage's result (e.g. the metrics of `validate`/`test`, the predictions of `predict`).
- Modules and trainers from the `lightning` package (`lightning.pytorch`) are accepted. Mixing them with
  `pytorch_lightning` objects in one run is reported as an error instead of failing inside Lightning.
- When a `_target_` names your own code without `project.` (or no `__lighter__.py` was found), the error says
  what to write instead.
- `--print_config` (or `--print-config`) prints the composed config, as YAML like the saved `config.yaml`, and exits
  without building anything. `lighter --version` prints the version.
- `lighter <stage> --help` explains the override syntax (`key::sub=value`, `=key=...`, `~key`) and shell quoting.
- The CLI prints a two-line banner: project folder, configs and number of overrides, seed, log folder.

- LighterModule's `optimizer` may be a function of the parameters and `scheduler` a function of the optimizer
  (`_mode_: callable` in the config), so configs no longer need `params: "$@model::network.parameters()"`.
  `scheduler` may also be Lightning's scheduler dict (`scheduler`, `monitor`, `interval`, ...).
- A LighterModule with only a `training_step` and a LighterDataModule with only a `train_dataloader` can `fit`.
  A stage whose dataloader is not set (e.g. `validate` without `val_dataloader`) names the key to add.

- `lighter.load_model(ckpt_path, overrides=())` rebuilds a model from the config saved in its checkpoint and
  loads its weights (with `torch.load(weights_only=True)` by default), finding the project from the checkpoint's folder.

### Changed

- Lighter now depends on six packages: torch, pytorch-lightning, torchmetrics, sparkwheel, pyyaml and cloudpickle.
  numpy, pandas, requests, loguru, rich, tensorboard and torchvision are no longer installed with it.
  Migration: `pip install tensorboard` to keep TensorBoard as Lightning's default logger (otherwise it logs CSV),
  `pip install torchvision` for FileWriter's `image_2d`/`image_3d` writers.
- `__lighter__.py` is searched for in each config file's folder and its parents first, then in the current
  directory and its parents, so `lighter fit path/to/project/configs/train.yaml` works from any folder.
  Configs from two different projects in one run are an error. The project folder no longer needs an `__init__.py`.
- `python -m lighter` no longer imports modules from the current directory; import your code as `project.<module>`.
- Config and build errors in the CLI print `file:line (key): Type: message`, the config lines, the hint and the
  overrides of that key, without a framework traceback, and exit with code 2. A bug in your own code shows only
  your frames and exits with 1. Errors raised inside `trainer.<stage>()` propagate unchanged. `LIGHTER_DEBUG=1`
  shows the full traceback.
- A missing config file is reported before anything is built, with a hint for files separated by commas,
  a forgotten `=` in an override, or a similar file name.
- `import lighter` no longer imports torch or Lightning, so `lighter --help` and `--print_config` start instantly.
- The boolean CLI flags (`--weights_only`, `--verbose`, `--return_predictions`) also accept a `--no-` form.

- LighterModule no longer defines stub step methods. As in Lightning, validation is skipped without a
  `validation_step`, and `validation_step`/`test_step` may return None (only the metrics are logged then).
- `loss` must be one tensor. Migration: instead of `{"loss": {"total": total, "ce": ce}}`, return
  `{"loss": total}` and log the parts with `self.log("train/loss/ce", ce)`; `train/loss/total/*` becomes `train/loss/*`.
- LighterModule logs losses and metrics also with `logger: false`, so callbacks such as `ModelCheckpoint` and
  `EarlyStopping` can monitor them.
- One `val_metrics`/`test_metrics` object used with several dataloaders is an error (it mixed their results).
- LighterDataModule returns `[]` instead of None for a dataloader that is not set.

- The composed config (with the seed and `_imports_`) is saved once the log folder is final, by the global-zero
  process only (also under DDP): `config.yaml` in the log folder, the logger's hyperparameters, and every
  checkpoint under `"lighter_config"`. It is no longer merged into the model's hyperparameters.
  Migration: read `checkpoint["lighter_config"]` instead of `checkpoint["hyper_parameters"]["config"]`,
  or use `lighter.load_model`.
- `--ckpt_path last` without a last checkpoint in the run's checkpoint folder is an error (Lightning silently
  started from scratch, since each run logs to a new version folder); the message names the latest checkpoint.
- Freezer only changes the parameters it selects and gives them back their own `requires_grad` when it
  unfreezes; a selection that matches no parameter is an error. Migration: Freezer no longer sets
  `requires_grad=True` on the other parameters; set it yourself if you relied on that.

### Removed

- `import lighter` no longer replaces the root logging handlers, installs a rich traceback hook or overrides
  `warnings.showwarning`. Lighter logs through the standard `logging` module (logger name `lighter`).
- LighterModule no longer logs optimizer statistics (`train/optimizer/...`).
  Migration: add `pytorch_lightning.callbacks.LearningRateMonitor` to `trainer::callbacks`.
- `lighter.utils.misc`, `lighter.utils.data` (`collate_replace_corrupted`), `lighter.utils.logging` and
  `lighter.utils.types.enums` are gone. Migration: pass stage names as strings, e.g. `Runner().run("fit", inputs)`.
- `writer_registry` is replaced by the `WRITERS` dict in `lighter.callbacks.file_writer`.
  Migration: pass a callable as `writer_fn` for a custom writer.
- Prediction writers no longer empty Lightning's private list of predictions after each batch.
  Migration: `lighter predict ... --no-return_predictions` keeps memory flat when the writer is all you need.

- `LighterModule.mode` and the private logging helpers (`_log_outputs`, `_log_loss`, `_log_metrics`, `_log`,
  `_normalize_output`). Migration: use `self.trainer.training` (or `validating`, `testing`, `predicting`);
  override `on_train_batch_end` (or the val/test hook) to change what is logged.

### Fixed

- The train loss LighterModule logs with `accumulate_grad_batches > 1` is the loss your step returned
  (it was divided by the number of accumulated batches).
- CsvWriter keeps every value exactly as written (IDs like `00001` no longer become `1`, `NA` is no longer
  read as missing) and replaces the final file in one step, so a failed merge never leaves a partial file.
- `numpy` is no longer pinned below 2.0, so Lighter installs next to NumPy 2.
- Sparkwheel is capped at `<0.1` until Lighter is tested against its next release; the minimum is 0.0.11.
- `pyyaml` is declared as a dependency (Lighter imports it directly).
- Project URLs point at `github.com/project-lighter/lighter`.
- The docs passed several config files separated by commas; the CLI takes them separated by spaces.
