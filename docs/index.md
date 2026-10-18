---
title: Lighter
---

# Lighter

--8<-- "README.md:intro"

```python title="model.py"
--8<-- "examples/quickstart/model.py"
```

```yaml title="config.yaml"
--8<-- "examples/quickstart/config.yaml"
```

--8<-- "README.md:run"

Train it yourself; the run takes seconds on a CPU and downloads no data:

--8<-- "README.md:try"

Lighter adds no training API of its own: the config builds Lightning objects, and Lightning runs them.

## Read next

| Page | For |
|------|-----|
| [Quickstart](quickstart.md) | The example above, file by file |
| [How a run works](run.md) | What `lighter fit` does, and what a run leaves behind |
| [Config cheat sheet](config.md) | `_target_`, `@`, `%`, `$`, overrides and shell quoting |
| [Your code and `project.`](your-code.md) | How Lighter imports your files |
| [LighterModule](lighter-module.md) | The optional base class that wires the optimizer, loss and metrics |
| [Recipes](recipes.md) | Variants, resuming, sweeps, loggers, several GPUs, models in a notebook |
| [Troubleshooting](troubleshooting.md) | What a mistake prints, and how to fix it |
| [Coming from LightningCLI or Hydra](coming-from.md) | The same ideas, in Lighter's words |
| [CLI](reference/cli.md) and [API](reference/index.md) reference | Every flag and every function |
