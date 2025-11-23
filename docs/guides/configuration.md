---
title: Configuration Guide
---

# Configuration Guide

Master Lighter's configuration system in 15 minutes.

Lighter uses [Sparkwheel](https://project-lighter.github.io/sparkwheel/) for configuration - a powerful YAML-based system with references, expressions, and object instantiation.

## The 5 Essential Symbols

You only need to understand 5 symbols to use Lighter effectively:

### 1. `_target_`: Create Objects

Instantiate any Python class from YAML:

```yaml
model:
  _target_: torch.nn.Linear
  in_features: 784
  out_features: 10
```

**Equivalent Python:**
```python
model = torch.nn.Linear(in_features=784, out_features=10)
```

Works with **any** Python class - PyTorch, third-party libraries, or your own code.

### 2. `@`: Resolved References (Lazy)

Reference values that are **resolved lazily** when needed:

```yaml
model:
  network:
    _target_: torchvision.models.resnet18
    num_classes: 10

  optimizer:
    _target_: torch.optim.Adam
    params: "$@model::network.parameters()"  # Resolved lazily
    lr: 0.001
```

**What `@` does:**
- Resolves **lazily** (when you call `resolve()`, not when loading config)
- Returns **final computed values** after instantiation and evaluation
- Gets the actual Python object, so you can call methods on it

### 3. `%`: Raw References (Eager)

Copy **raw YAML content** that is processed **eagerly** during config merge:

```yaml
model:
  train_metrics:
    - _target_: torchmetrics.Accuracy
      task: multiclass
      num_classes: 10

  val_metrics: "%model::train_metrics"  # Copies raw YAML
```

**What `%` does:**
- Processes **eagerly** (during config loading, before instantiation)
- Copies **unprocessed YAML** definition
- Creates a **new instance** when later resolved (not shared!)

!!! danger "Critical: Use `%` for Metrics, Not `@`"
    Metrics accumulate state. Using `@` shares the same instance between train and val:

    ```yaml
    # ❌ WRONG - Shares the same metric instance
    val_metrics: "@model::train_metrics"

    # ✅ CORRECT - Copies config, creates separate instance
    val_metrics: "%model::train_metrics"
    ```

    **Why this matters:** `%` copies the raw YAML template, so when each is resolved, you get separate instances. `@` would resolve once and share that same object.

### 4. `$`: Evaluate Python Expressions

Run Python code in your configs:

```yaml
# Simple math
lr: "$0.001 * 2"  # = 0.002

# Call methods
optimizer:
  params: "$@model::network.parameters()"

# Conditionals
batch_size: "$64 if %vars::large_batch else 32"

# List comprehensions
layer_sizes: "$[64 * (2**i) for i in range(4)]"  # [64, 128, 256, 512]

# Type conversions
warmup_steps: "$int(%vars::total_steps * 0.1)"
```

### 5. `::`: Navigate Config Paths

Access nested values using `::` separator:

```yaml
model::optimizer::lr           # Navigate to nested value
data::train_dataloader::batch_size
```

Use in CLI overrides:
```bash
lighter fit config.yaml model::optimizer::lr=0.01
```

## The Critical Rule: `::` vs `.`

- `::` navigates **config** structure
- `.` accesses **Python** attributes/methods

```yaml
# ❌ WRONG
params: "$@model::network::parameters()"  # :: for Python method

# ✅ CORRECT
params: "$@model::network.parameters()"   # . for Python method
```

## Common Patterns

### Pattern 1: Network → Optimizer

```yaml
model:
  network:
    _target_: torchvision.models.resnet50
    num_classes: 10

  optimizer:
    _target_: torch.optim.Adam
    params: "$@model::network.parameters()"  # Pass network params
    lr: 0.001
```

### Pattern 2: Optimizer → Scheduler

```yaml
model:
  optimizer:
    _target_: torch.optim.Adam
    params: "$@model::network.parameters()"
    lr: 0.001

  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    optimizer: "@model::optimizer"  # Pass optimizer object
    T_max: 100
```

### Pattern 3: Reusing Metrics

```yaml
model:
  train_metrics:
    - _target_: torchmetrics.Accuracy
      task: multiclass
      num_classes: 10
    - _target_: torchmetrics.F1Score
      task: multiclass
      num_classes: 10

  val_metrics: "%model::train_metrics"  # Reuse config
  test_metrics: "%model::train_metrics"
```

### Pattern 4: Shared Variables

```yaml
vars:
  num_classes: 10
  base_lr: 0.001
  batch_size: 32

model:
  network:
    _target_: torchvision.models.resnet18
    num_classes: "%vars::num_classes"

  optimizer:
    lr: "%vars::base_lr"

data:
  train_dataloader:
    batch_size: "%vars::batch_size"
```

### Pattern 5: Differential Learning Rates

```yaml
model:
  optimizer:
    _target_: torch.optim.SGD
    params:
      - params: "$@model::network.backbone.parameters()"
        lr: 0.0001  # Low LR for pretrained backbone
      - params: "$@model::network.head.parameters()"
        lr: 0.01    # High LR for new head
    momentum: 0.9
```

## Config Structure

Every Lighter config has three main sections:

```yaml
trainer:  # PyTorch Lightning Trainer
  _target_: pytorch_lightning.Trainer
  max_epochs: 10
  accelerator: auto
  devices: 1

model:    # LightningModule or LighterModule
  _target_: your.Module
  # ... module arguments ...

data:     # LighterDataModule or custom LightningDataModule
  _target_: lighter.LighterDataModule
  train_dataloader: ...
  val_dataloader: ...
```

### Optional Sections

```yaml
_requires_:  # Import Python modules
  - "$import torch"
  - "$from datetime import datetime"

vars:        # Reusable variables
  num_classes: 10
  lr: 0.001

args:        # Stage-specific arguments
  fit:
    ckpt_path: null
  test:
    ckpt_path: "checkpoints/best.ckpt"
```

## CLI Overrides

Override any config value from command line:

```bash
# Single override
lighter fit config.yaml trainer::max_epochs=100

# Nested values
lighter fit config.yaml model::optimizer::lr=0.001

# Multiple overrides
lighter fit config.yaml \
  trainer::max_epochs=100 \
  model::optimizer::lr=0.001 \
  data::train_dataloader::batch_size=64
```

## Merging Configs

Combine multiple YAML files:

```bash
lighter fit base.yaml,experiment.yaml
```

### Default Behavior: Merge

Configs merge automatically:

```yaml
# base.yaml
trainer:
  max_epochs: 10
  accelerator: auto

# experiment.yaml
trainer:
  max_epochs: 100  # Overrides
  devices: 4       # Adds
```

**Result:** `max_epochs=100`, `accelerator=auto`, `devices=4`

### Replace with `=`

Replace instead of merge:

```yaml
# experiment.yaml
trainer:
  =callbacks:  # Replace entire callbacks list
    - _target_: pytorch_lightning.callbacks.EarlyStopping
```

### Delete with `~`

Remove keys:

```yaml
# Delete entire key
trainer:
  ~callbacks: null

# Delete list items by index
trainer:
  ~callbacks: [1, 3]  # Remove items at indices 1 and 3

# Delete dict keys
data:
  ~test_dataloader: null
```

## Common Pitfalls

### ❌ Wrong: Using `@` for Metrics

```yaml
val_metrics: "@model::train_metrics"  # Shared instance!
```

### ✅ Correct: Using `%` for Metrics

```yaml
val_metrics: "%model::train_metrics"  # New instance
```

---

### ❌ Wrong: Using `::` for Python Attributes

```yaml
params: "$@model::network::parameters()"
```

### ✅ Correct: Using `.` for Python Attributes

```yaml
params: "$@model::network.parameters()"
```

---

### ❌ Wrong: Missing `$` for Expressions

```yaml
batch_size: "@vars::base_batch * 2"  # Treated as string!
```

### ✅ Correct: Using `$` for Expressions

```yaml
batch_size: "$%vars::base_batch * 2"  # Evaluated
```

## Advanced: Conditional Config

```yaml
vars:
  use_pretrained: true

model:
  network:
    _target_: torchvision.models.resnet18
    weights: "$'IMAGENET1K_V2' if %vars::use_pretrained else None"
    num_classes: 10
```

## Advanced: Dynamic Imports

```yaml
_requires_:
  - "$import datetime"
  - "$from pathlib import Path"

trainer:
  logger:
    name: "$datetime.datetime.now().strftime('%Y%m%d_%H%M%S')"
```

## Complete Example

```yaml
_requires_:
  - "$import torch"

vars:
  num_classes: 10
  base_lr: 0.001
  max_epochs: 100

trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: "%vars::max_epochs"
  accelerator: auto
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      monitor: val_loss
      mode: min
      save_top_k: 3

model:
  _target_: lighter.LighterModule

  network:
    _target_: torchvision.models.resnet18
    num_classes: "%vars::num_classes"

  criterion:
    _target_: torch.nn.CrossEntropyLoss

  optimizer:
    _target_: torch.optim.Adam
    params: "$@model::network.parameters()"
    lr: "%vars::base_lr"

  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    optimizer: "@model::optimizer"
    T_max: "%vars::max_epochs"

  train_metrics:
    - _target_: torchmetrics.Accuracy
      task: multiclass
      num_classes: "%vars::num_classes"

  val_metrics: "%model::train_metrics"

data:
  _target_: lighter.LighterDataModule
  train_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 32
    shuffle: true
    dataset:
      _target_: torchvision.datasets.CIFAR10
      root: ./data
      train: true
      download: true
```

## Quick Reference

| Symbol | Use | Example |
|--------|-----|---------|
| `_target_` | Instantiate class | `_target_: torch.nn.Linear` |
| `@` | Resolved reference | `@model::optimizer` |
| `%` | Raw reference | `%model::train_metrics` |
| `$` | Python expression | `$0.001 * 2` |
| `::` | Config path | `model::optimizer::lr` |
| `.` | Python attribute | `@model::network.parameters()` |
| `=` | Replace operator | `=callbacks:` |
| `~` | Delete operator | `~callbacks: [0, 2]` |

## Next Steps

- [Custom Code Guide](custom-code.md) - Use your own models/datasets
- [Training Guide](training.md) - Run experiments
- [Sparkwheel Docs](https://project-lighter.github.io/sparkwheel/) - Complete reference
