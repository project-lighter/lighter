---
title: Quick Start
---

# Quick Start

Get a model training in 10 minutes.

## 1. Install

```bash
pip install lighter
```

## 2. Create Your Project

Lighter uses a **project folder** pattern for organizing your code. This makes it easy to reference your custom models and datasets.

### Step 1: Create the Project Structure

```bash
mkdir mnist_classifier
cd mnist_classifier
```

Create these files:

```
mnist_classifier/
├── __lighter__.py          # Marker file (tells Lighter this is a project)
├── __init__.py             # Makes it a Python package
├── model.py                # Your model code
└── configs/
    └── config.yaml         # Your experiment config
```

### Step 2: Add the Marker Files

**`__lighter__.py`** (can be empty):
```python
# This file marks the directory as a Lighter project.
# When you run `lighter` from this directory, you can reference
# your code as `project.module.Class`
```

**`__init__.py`** (can be empty):
```python
# Makes this directory a Python package
```

!!! tip "The `project` Prefix"
    Once you have `__lighter__.py`, Lighter auto-discovers your folder and makes it available as `project`. This means `model.py` becomes `project.model`, and you can reference classes as `project.model.ClassName`.

## 3. Choose Your Approach

Lighter works with **any** PyTorch Lightning module. Pick the style that fits your workflow:

- **LightningModule** - Use existing Lightning code, add configs
- **LighterModule** - Less boilerplate, automatic logging

=== "LightningModule"

    ### Write Your Module

    **`model.py`**:

    ```python
    import pytorch_lightning as pl
    import torch
    import torch.nn.functional as F

    class MNISTModule(pl.LightningModule):
        def __init__(self, learning_rate=0.001):
            super().__init__()
            self.lr = learning_rate
            self.model = torch.nn.Sequential(
                torch.nn.Linear(28 * 28, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 10)
            )

        def forward(self, x):
            return self.model(x.view(x.size(0), -1))

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = F.cross_entropy(self(x), y)
            self.log("train/loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            loss = F.cross_entropy(self(x), y)
            acc = (self(x).argmax(dim=1) == y).float().mean()
            self.log("val/loss", loss)
            self.log("val/acc", acc)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)
    ```

    ### Create Config

    **`configs/config.yaml`**:

    ```yaml
    trainer:
      _target_: pytorch_lightning.Trainer
      max_epochs: 3
      accelerator: auto

    model:
      _target_: project.model.MNISTModule  # project.file.Class
      learning_rate: 0.001

    data:
      _target_: lighter.LighterDataModule
      train_dataloader:
        _target_: torch.utils.data.DataLoader
        batch_size: 64
        shuffle: true
        dataset:
          _target_: torchvision.datasets.MNIST
          root: ./data
          train: true
          download: true
          transform:
            _target_: torchvision.transforms.ToTensor

      val_dataloader:
        _target_: torch.utils.data.DataLoader
        batch_size: 64
        dataset:
          _target_: torchvision.datasets.MNIST
          root: ./data
          train: false
          download: true
          transform:
            _target_: torchvision.transforms.ToTensor
    ```

    ### Run

    ```bash
    lighter fit configs/config.yaml
    ```

    **That's it!** You now have:

    - ✅ Training and validation loops
    - ✅ Automatic logging
    - ✅ Checkpointing
    - ✅ Progress bars

=== "LighterModule"

    ### Write Your Module

    **`model.py`**:

    ```python
    from lighter import LighterModule
    import torch

    class MNISTModel(LighterModule):
        def training_step(self, batch, batch_idx):
            x, y = batch
            x = x.view(x.size(0), -1)  # Flatten
            pred = self(x)
            loss = self.criterion(pred, y)

            if self.train_metrics:
                self.train_metrics(pred, y)

            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            x, y = batch
            x = x.view(x.size(0), -1)  # Flatten
            pred = self(x)
            loss = self.criterion(pred, y)

            if self.val_metrics:
                self.val_metrics(pred, y)

            return {"loss": loss}
    ```

    ### Create Config

    **`configs/config.yaml`**:

    ```yaml
    trainer:
      _target_: pytorch_lightning.Trainer
      max_epochs: 3
      accelerator: auto

    model:
      _target_: project.model.MNISTModel  # project.file.Class

      network:
        _target_: torch.nn.Sequential
        _args_:
          - _target_: torch.nn.Linear
            in_features: 784
            out_features: 128
          - _target_: torch.nn.ReLU
          - _target_: torch.nn.Linear
            in_features: 128
            out_features: 10

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
        batch_size: 64
        shuffle: true
        dataset:
          _target_: torchvision.datasets.MNIST
          root: ./data
          train: true
          download: true
          transform:
            _target_: torchvision.transforms.ToTensor

      val_dataloader:
        _target_: torch.utils.data.DataLoader
        batch_size: 64
        dataset:
          _target_: torchvision.datasets.MNIST
          root: ./data
          train: false
          download: true
          transform:
            _target_: torchvision.transforms.ToTensor
    ```

    ### Run

    ```bash
    lighter fit configs/config.yaml
    ```

    **That's it!** You get everything from the LightningModule approach PLUS:

    - ✅ Automatic `configure_optimizers()`
    - ✅ Dual logging (step + epoch)
    - ✅ Config-driven metrics

## 4. Experiment!

Now that it's running, try changing things from the CLI:

```bash
# Change learning rate
lighter fit configs/config.yaml model::lr=0.01

# Train longer
lighter fit configs/config.yaml trainer::max_epochs=10

# Larger batch size
lighter fit configs/config.yaml data::train_dataloader::batch_size=128

# Combine multiple changes
lighter fit configs/config.yaml \
  model::lr=0.01 \
  trainer::max_epochs=10 \
  data::train_dataloader::batch_size=128
```

No file editing needed!

## Understanding the Config

Let's break down what's happening:

### The Three Keys

Every Lighter config has three main sections:

```yaml
trainer:  # How to run (PyTorch Lightning Trainer)
model:    # What to run (your LightningModule)
data:     # What data to use (DataModule)
```

### The `_target_` Pattern

`_target_` tells Lighter what class to instantiate:

```yaml
model:
  _target_: project.model.MNISTModule  # Create instance of this class
  learning_rate: 0.001                  # Pass as __init__ argument
```

This is equivalent to:

```python
model = MNISTModule(learning_rate=0.001)
```

### The `project.` Prefix

When you have `__lighter__.py` in your folder, Lighter auto-discovers it and makes it available as `project`:

```yaml
# Your file: model.py
# Your class: MNISTModule
# Reference as: project.model.MNISTModule

_target_: project.model.MNISTModule
```

This pattern keeps all your custom code organized and easy to reference.

### References

Use `@` to reference resolved values (after instantiation):

```yaml
optimizer:
  params: "$@model::network.parameters()"  # Call method on instantiated network
```

Use `%` to copy config (for creating new instances):

```yaml
val_metrics: "%model::train_metrics"  # New instance with same config
```

⚠️ **Important:** Use `%` not `@` for metrics (they're stateful and need separate instances)

## What You Just Learned

- ✅ How to create a Lighter project with `__lighter__.py`
- ✅ How to reference your code as `project.*`
- ✅ Two ways to use Lighter (LightningModule or LighterModule)
- ✅ Basic config structure (trainer/model/data)
- ✅ How to run experiments
- ✅ How to override from CLI
- ✅ Key config syntax (`_target_`, `@`, `%`)

## Next Steps

### Learn More Config Syntax

[Configuration Guide](guides/configuration.md) - Master Sparkwheel syntax

### See Complete Examples

[Example Projects](examples/index.md) - Complete examples across various domains (image classification, medical imaging, NLP, and more)

### Organize Your Project

[Custom Code Guide](guides/custom-code.md) - Best practices for structuring larger projects

### Best Practices

[Best Practices](guides/best-practices.md) - Production patterns

## Common Next Questions

**Q: How do I add custom datasets or transforms?**
A: Put them in separate files (e.g., `data.py`, `transforms.py`) and reference as `project.data.MyDataset` or `project.transforms.MyTransform`

**Q: Can I use multiple GPUs?**
A: Yes! Just add `trainer::devices=-1` (all GPUs) or `trainer::devices=4` (4 GPUs)

**Q: How do I save predictions?**
A: Use Writers - see [Training Guide](guides/training.md#saving-predictions)

**Q: Something not working?**
A: Check [FAQ](faq.md) or [join Discord](https://discord.gg/zJcnp6KrUp)

## Quick Reference

```yaml
# Essential config structure
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10

model:
  _target_: project.model.MyModule  # Your Lightning module
  # ... your module's __init__ args ...

data:
  _target_: lighter.LighterDataModule
  train_dataloader: ...
  val_dataloader: ...
```

```bash
# Essential CLI commands
lighter fit config.yaml                  # Train
lighter validate config.yaml             # Validate only
lighter test config.yaml                 # Test only
lighter predict config.yaml              # Inference

# Override any config value
lighter fit config.yaml key::path=value
```

[→ Full CLI Reference](reference/cli.md)
