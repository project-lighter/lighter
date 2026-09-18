---
title: Lighter
toc_depth: 1
---

<!-- Fake title -->
#

<style>
    /* Remove content from the left bar (otherwise there's "Home" just sitting there) */
    .md-nav--primary {
    display: none;
    }
</style>


<!-- Logo -->
<div style="display: flex; justify-content: center;"><img src="assets/images/lighter_banner.png" style="width:65%;"/></div>

<!-- pip install -->
<div style="width:65%; margin:auto; text-align:center">
</br>

This working development pair requires matching Lighter and Sparkwheel builds. Follow [compatibility and local installation](guides/compatibility.md).

<!-- [![PyPI](https://img.shields.io/pypi/v/lighter)](https://pypi.org/project/lighter/)
[![Python](https://img.shields.io/pypi/pyversions/lighter)](https://pypi.org/project/lighter/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://img.shields.io/discord/1234567890?label=Discord&logo=discord)](https://discord.gg/zJcnp6KrUp) -->

</div>
</br>

**YAML configuration for PyTorch Lightning experiments**

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Fast Iteration**

    ---

    Change hyperparameters from CLI without editing code.

    ```bash
    lighter fit config.yaml model::learning_rate=0.01
    ```

-   :material-refresh:{ .lg .middle } **Reproducible**

    ---

    Version recipes alongside code, data identities, dependencies and recorded runtime evidence.

-   :material-lightning-bolt:{ .lg .middle } **Pure Lightning**

    ---

    Use native LightningModules with their own hooks and optimization.

</div>

## What is Lighter?

Lighter composes PyTorch Lightning experiments from YAML recipes. Scientific behavior remains ordinary Python.

**You write Lightning code. Lighter handles configuration.**

```python title="model.py"
import pytorch_lightning as pl


class MyModule(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.lr = learning_rate
        # ... your model code ...

    def training_step(self, batch, batch_idx):
        # ... your training logic ...
        return loss
```

```yaml title="config.yaml"
model:
  _target_: project.model.MyModule  # Auto-discovered with __lighter__.py
  learning_rate: 0.001

trainer:
  max_epochs: 10
```

```bash
# Run it
lighter fit config.yaml

# Override from CLI
lighter fit config.yaml model::learning_rate=0.01
```

## Two Approaches, Same Power

Choose the approach that fits your workflow:

<div class="grid" markdown>

<div markdown>

### :material-code-braces: LightningModule

**Best for:**

- Existing Lightning projects
- Custom training logic
- Full control over everything

**You write:**

- All step methods
- `configure_optimizers()`
- Your own logging

**Lighter adds:**

- YAML configuration
- CLI overrides
- Experiment tracking

[Learn more →](guides/lightning-module.md)

</div>

<div markdown>

### :material-auto-fix: LighterModule

**Best for:**

- New projects
- Standard workflows
- Less boilerplate

**You write:**

- Step implementations only
- Your model's forward logic

**Lighter adds:**

- Automatic `configure_optimizers()`
- Dual logging (step + epoch)
- Configured objects and dependencies

[Learn more →](guides/lighter-module.md)

</div>

</div>

!!! tip "Choose ownership explicitly"
    Both approaches use the same configuration system. Native LightningModules own their hooks and optimization; LighterModule supplies selected conveniences. Moving between them may require adapting constructors, logging or optimizer ownership. Changing `_target_` alone does not transform arbitrary module code.

## Quick Comparison

=== "LightningModule"

    ```python title="model.py"
    import torch
    import torch.nn.functional as F
    import pytorch_lightning as pl


    class MyModule(pl.LightningModule):
        def __init__(self, network, learning_rate=0.001):
            super().__init__()
            self.network = network
            self.lr = learning_rate

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = F.cross_entropy(self.network(x), y)
            self.log("train/loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)
    ```

    ```yaml title="config.yaml"
    trainer:
      _target_: pytorch_lightning.Trainer
      max_epochs: 10

    model:
      _target_: project.model.MyModule  # project.file.Class
      network:
        _target_: torchvision.models.resnet18
        num_classes: 10
      learning_rate: 0.001

    data:
      _target_: lighter.LighterDataModule
      train_dataloader:
        _target_: torch.utils.data.DataLoader
        batch_size: 32
        dataset:
          _target_: torchvision.datasets.CIFAR10
          root: ./data
          train: true
          download: true
          transform:
            _target_: torchvision.transforms.ToTensor
    ```

    ```bash
    lighter fit config.yaml
    ```

=== "LighterModule"

    ```python title="model.py"
    from lighter import LighterModule


    class MyModel(LighterModule):
        def training_step(self, batch, batch_idx):
            x, y = batch
            pred = self(x)
            loss = self.criterion(pred, y)

            if self.train_metrics:
                self.train_metrics(pred, y)

            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            x, y = batch
            pred = self(x)
            loss = self.criterion(pred, y)

            if self.val_metrics:
                self.val_metrics(pred, y)

            return {"loss": loss}
    ```

    ```yaml title="config.yaml"
    trainer:
      _target_: pytorch_lightning.Trainer
      max_epochs: 10

    model:
      _target_: project.model.MyModel  # project.file.Class
      network:
        _target_: torchvision.models.resnet18
        num_classes: 10
      criterion:
        _target_: torch.nn.CrossEntropyLoss
      optimizer:
        _target_: torch.optim.Adam
        params: "$@model::network.parameters()"
        lr: 0.001
      train_metrics:
        - _target_: torchmetrics.Accuracy
          task: multiclass
          num_classes: 10
      val_metrics: "%model::train_metrics"

    data:
      _target_: lighter.LighterDataModule
      train_dataloader:
        _target_: torch.utils.data.DataLoader
        batch_size: 32
        dataset:
          _target_: torchvision.datasets.CIFAR10
          root: ./data
          train: true
          download: true
          transform:
            _target_: torchvision.transforms.ToTensor
    ```

    ```bash
    lighter fit config.yaml
    ```

## Why Lighter?

### Reproducibility

Version and compare the intended recipe, then retain code, data, environment, effective settings and artifact identities. A configuration diff alone does not establish equivalent execution.

```bash
git diff experiment_v1.yaml experiment_v2.yaml
```

See exactly what changed between experiments.

### Fast Iteration

Override any config value from CLI:

```bash
# Change learning rate
lighter fit config.yaml model::learning_rate=0.01

# Use more GPUs
lighter fit config.yaml trainer::devices=4

# Combine multiple changes
lighter fit config.yaml model::learning_rate=0.01 trainer::max_epochs=100
```

### No Lock-In

Lighter is a thin layer over PyTorch Lightning:

- Keep native LightningModule hooks and custom optimization
- Compose compatible native callbacks and loggers
- Inspect the native objects and observed runtime state
- Use an explicit native implementation as a scientific control

## Installation

This working development pair requires matching Lighter and Sparkwheel builds. Follow [compatibility and local installation](guides/compatibility.md).

## Get Started

Ready to try it? Pick your path:

<div class="grid cards" markdown>

-   :material-rocket:{ .lg .middle } **Quick Start**

    ---

    Get a model training in 10 minutes.

    [:octicons-arrow-right-24: Quick Start](quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **Complete Examples**

    ---

    One research walkthrough, a diagnostic and explicit integration status.

    [:octicons-arrow-right-24: Examples](examples/index.md)

-   :material-school:{ .lg .middle } **Guides**

    ---

    Task-focused how-to guides.

    [:octicons-arrow-right-24: Guides](guides/configuration.md)

</div>

## Example Projects

Start with [Compare and Continue](https://github.com/project-lighter/lighter/tree/main/projects/experiment_comparison) for an actual research decision, matched native controls, selected-checkpoint predictions and restored-state continuation. Use the [quick start](quickstart.md) for a download-free diagnostic.

The [project guide](examples/index.md) explains the scope and status of older domain integrations. Lighter is general purpose; each new scientific task still needs its own data, objective and artifact checks.

## Community

- [:fontawesome-brands-discord: Discord](https://discord.gg/zJcnp6KrUp) - Get help, share configs
- [:fontawesome-brands-github: GitHub](https://github.com/project-lighter/lighter) - Report issues, contribute
- [:material-file-document: Paper](https://joss.theoj.org/papers/10.21105/joss.08101) - Cite us

## What Next?

- [**Quick Start** - 10 minutes to running model](quickstart.md)
- [**Configuration Guide** - Learn the syntax](guides/configuration.md)
- [**FAQ** - Common questions](faq.md)
