---
title: How a run works
---

# How a run works

`lighter fit config.yaml model::lr=0.001` does six things, in this order.

1. **Compose the config.** The inputs are applied left to right: each YAML file is merged into the ones before
   it, then each override is applied. Dicts merge key by key, lists are extended, `=key` replaces and `~key`
   deletes (see the [config cheat sheet](config.md)). `--print_config` prints the result and stops here.
2. **Seed.** The seed is the top-level `seed:` key, else the `PL_GLOBAL_SEED` environment variable, else 0.
   Lighter calls Lightning's `seed_everything(seed, workers=True)` and writes the seed into the config.
3. **Import your code.** Lighter looks for `__lighter__.py` in each config file's folder and its parents, then in
   the current folder and its parents, and imports the folder that holds it as the package `project`
   (see [Your code and `project.`](your-code.md)).
4. **Build three objects.** `model:` must build a `LightningModule` and `trainer:` a `Trainer`. `data:` is
   optional and must build a `LightningDataModule`; without it, the model's own dataloader methods are used.
   They may come from `pytorch_lightning` or `lightning.pytorch`, but not from both in one run.
5. **Add one callback.** Lighter appends a callback to the trainer that saves the composed config with the run.
6. **Run the stage.** Lighter calls `trainer.fit(model, datamodule=data)`, passing on `--ckpt_path` and the
   stage's other flags. From here on it is Lightning.

The same is available from Python. `lighter.build` does steps 1 to 5 and returns the three objects:

```python
import lighter

model, trainer, data = lighter.build("config.yaml", "model::lr=0.001")
trainer.fit(model, datamodule=data)
```

## What a run leaves behind

Lightning gives every run, of any stage, a new folder under `lightning_logs/` (or under your logger's
`save_dir`). Lighter adds `config.yaml` to it:

```text
lightning_logs/
└── version_0/
    ├── config.yaml        # the composed config, including the seed
    ├── hparams.yaml       # the same config, logged as hyperparameters
    ├── metrics.csv        # or TensorBoard files, if tensorboard is installed
    └── checkpoints/
        └── epoch=19-step=160.ckpt
```

Every checkpoint also holds the config, under the key `"lighter_config"`, except one saved with
`save_weights_only=True`. `lighter.load_model` rebuilds the model from that config, or from the run's
`config.yaml`, and loads the weights (see [`load_model`][lighter.engine.saved_config.load_model]). With
`trainer::logger=false` there is no log folder, so no `config.yaml`; checkpoints still hold the config.

## What Lighter prints

Before the stage starts, Lighter prints two lines: the project folder, the configs and number of overrides,
the seed, and the log folder. Everything after that comes from Lightning.

When the config cannot be built, Lighter prints the file, line and key, the lines around it and a hint, and
exits with code 2. When the error comes from your own code while it is being built (for example a model's
`__init__`), it shows only your frames and exits with 1. Errors raised during the stage itself, such as in
`training_step`, are Lightning's and pass through unchanged. `LIGHTER_DEBUG=1` shows full tracebacks.
[Troubleshooting](troubleshooting.md) shows what each common mistake prints.
