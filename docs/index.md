---
title: Lighter
---

# Lighter

Lighter runs your PyTorch Lightning code from YAML configs. You write the LightningModule; a config says how to
build it, the trainer and the data; any value can be changed from the command line.

```bash
pip install lighter
git clone --depth 1 https://github.com/project-lighter/lighter
lighter fit lighter/examples/quickstart/config.yaml
```

Lighter is small and adds no API of its own to learn: Lightning is the API, and the YAML only
builds Python objects.

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
