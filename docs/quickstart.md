---
title: Quickstart
---

# Quickstart

Train a small network on random data, on a CPU, in under a minute:

```bash
pip install lighter
git clone --depth 1 https://github.com/project-lighter/lighter
lighter fit lighter/examples/quickstart/config.yaml
```

## What you ran

[`examples/quickstart`](https://github.com/project-lighter/lighter/tree/main/examples/quickstart) holds four files.
`model.py` is a plain LightningModule:

```python title="model.py"
--8<-- "examples/quickstart/model.py"
```

`data.py` is a dataset:

```python title="data.py"
--8<-- "examples/quickstart/data.py"
```

`config.yaml` says how to build the trainer, the model and the data. Each `_target_` names a class, and the
keys next to it are its arguments:

```yaml title="config.yaml"
--8<-- "examples/quickstart/config.yaml"
```

`__lighter__.py` marks the project folder. Lighter imports that folder as the package `project`, which is why the
config can say `project.model.Regressor`. The file's content does not matter.

Lighter built the three objects and called `trainer.fit(model, datamodule=data)`. Lightning wrote the run to
`lightning_logs/version_0/` (its metrics and a checkpoint), and Lighter saved the config it ran there as
`config.yaml`.

## Change it from the command line

Any value in the config can be overridden with `key::subkey=value`, without editing the file:

```bash
lighter fit lighter/examples/quickstart/config.yaml model::lr=0.001
lighter fit lighter/examples/quickstart/config.yaml model::hidden=64 trainer::max_epochs=20
```

Each run gets a new `version_N` folder. To see the composed config without running anything:

```bash
lighter fit lighter/examples/quickstart/config.yaml model::lr=0.001 --print_config
```

## Next

- What `lighter fit` did, step by step: [How a run works](run.md)
- The config syntax on one page: [Config cheat sheet](config.md)
- Use your own code: [Your code and `project.`](your-code.md)
