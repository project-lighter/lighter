# Tabular regression

A `LighterModule` that learns a known function, `y = 2 * x0 - 3 * x1 + 0.5`, from synthetic data. Train,
validation and test samples are disjoint, and each sample has an ID that predictions keep. It needs nothing
beyond Lighter, runs on a CPU in seconds, and CI runs it on every pull request.

```bash
cd projects/tabular_regression
lighter fit config.yaml
lighter test config.yaml --ckpt_path lightning_logs/version_0/checkpoints/epoch=19-step=160.ckpt
lighter predict config.yaml --ckpt_path lightning_logs/version_0/checkpoints/epoch=19-step=160.ckpt
```

The [LighterModule page](https://project-lighter.github.io/lighter/lighter-module/) walks through `task.py` and
`config.yaml`. `recipes/` holds the files of the [Recipes page](https://project-lighter.github.io/lighter/recipes/):
resuming, sweeps, writers, loggers and more, each run by a test.
