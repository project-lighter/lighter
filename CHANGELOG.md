# Changelog

All notable changes to Lighter are listed here. Breaking changes come with a one-line migration.

## Unreleased

### Added

- A top-level `seed:` key (default: 0) seeds the run, including DataLoader workers, and is written into the
  saved config; `seed=7` works without a `seed:` key in the config. Migration: rename any top-level `seed` key
  you used for something else.
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
- The CLI prints a two-line banner: project folder, configs and number of overrides, seed, log folder. After
  `fit` it prints the best checkpoint's path; `validate`, `test` and `predict` without `--ckpt_path` warn that
  they use the model as the config builds it.
- An error from your code that imports a folder of your project without `project.` (`from models import x`)
  says to write `from project.models import x`.

- LighterModule's `optimizer` may be a function of the parameters and `scheduler` a function of the optimizer
  (`_mode_: callable` in the config), so configs no longer need `params: "$@model::network.parameters()"`.
  `scheduler` may also be Lightning's scheduler dict (`scheduler`, `monitor`, `interval`, ...).
- A LighterModule with only a `training_step` and a LighterDataModule with only a `train_dataloader` can `fit`.
  A stage whose dataloader is not set (e.g. `validate` without `val_dataloader`) names the key to add.

- `lighter.load_model(ckpt_path, *overrides)` rebuilds a model from the config saved in its checkpoint and
  loads its weights (with `torch.load(weights_only=True)` by default), finding the project from the checkpoint's folder.
- A step that returns something other than a loss tensor, a number, a dict with a `"loss"` key or None
  (e.g. a tuple) warns that no loss is logged.

- `examples/quickstart`: a plain LightningModule, a dataset of random numbers and a config that train on a CPU in
  seconds, with no download. The README and the docs show these files, and a test runs them.
- The docs are ten short pages instead of ~7,000 lines that re-documented Lightning. A test builds every YAML block,
  runs every Python block and parses every `lighter` command in them, and keeps them under 3,000 lines.
- `projects/tabular_regression`: a download-free `LighterModule` example with disjoint train, validation and test
  samples, run by a test. Its `recipes/` folder holds the files of the Recipes page (variants, resuming, seeds,
  freezing, prediction writers, sweeps with a shell loop, Optuna or W&B, loggers, `load_model`), and a test runs
  every command on that page in order.

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
- Overrides may also follow options, e.g. `lighter fit config.yaml --ckpt_path last trainer::max_epochs=5`.
  A config file whose path contains `=` (`runs/lr=0.1/config.yaml`) is read as a file, not as an override.

- LighterModule no longer defines stub step methods. As in Lightning, validation is skipped without a
  `validation_step`, and `validation_step`/`test_step` may return None (only the metrics are logged then).
- `loss` must be one tensor. Migration: instead of `{"loss": {"total": total, "ce": ce}}`, return
  `{"loss": total}` and log the parts with `self.log("train/loss/ce", ce)`; `train/loss/total/*` becomes `train/loss/*`.
- LighterModule logs losses and metrics also with `logger: false`, so callbacks such as `ModelCheckpoint` and
  `EarlyStopping` can monitor them. It logs per-step values (`/step`) in training only, and shows
  `train/loss/step` and `val/loss/epoch` in the progress bar. Migration: monitor `val/.../epoch`, not `/step`.
- One `val_metrics`/`test_metrics` object used with several dataloaders is an error (it mixed their results).
- A dataloader not given to LighterDataModule counts as not defined, as if its method were missing, so Lightning
  skips that loop without a warning. Migration: check `datamodule.val_dataloader is None` instead of calling it.

- The composed config (with the seed and `_imports_`) is saved once the log folder is final, by the global-zero
  process only (also under DDP): `config.yaml` in the log folder, the logger's hyperparameters, and every
  checkpoint under `"lighter_config"`. It is no longer merged into the model's hyperparameters.
  Migration: read `checkpoint["lighter_config"]` instead of `checkpoint["hyper_parameters"]["config"]`,
  or use `lighter.load_model`.
- `--ckpt_path last` without a last checkpoint is an error (Lightning warned and started from scratch, and each
  run logs to a new version folder). `--ckpt_path best` is an error: it needs a `fit` in the same process.
  `--weights_only` on Lightning before 2.6 is reported instead of failing with a `TypeError`.
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
- CsvWriter ends lines with `\n` on every platform (Windows wrote `\r\n`).
- If `config.yaml` cannot be written because a dict config holds Python objects, the run warns and continues.
- The README states Lighter's real size (~1,200 lines); a test keeps `src/lighter` at most 1,400 lines and the README within 10%.
- `projects/cifar10` and `projects/lora` validated on the test set; they now hold out 10% of the training images
  for validation, chosen by a fixed `split_seed`, and keep the test set for `lighter test`. With
  `configs/fake_data.yaml`, CIFAR-10 runs every stage on random images without a download, and CI does so on
  every pull request. The example configs pass their optimizer and scheduler as `_mode_: callable`.
