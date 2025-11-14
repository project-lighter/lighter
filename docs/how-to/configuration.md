---
title: Configuration Reference
---

# Configuration Reference

Lighter uses **[Sparkwheel](https://project-lighter.github.io/sparkwheel/)** for configuration—a powerful YAML-based system supporting references, expressions, and object instantiation.

!!! tip "Complete Documentation"
    This page covers Lighter-specific patterns and common usage. For complete Sparkwheel syntax, advanced features, and detailed examples, see the **[Sparkwheel documentation](https://project-lighter.github.io/sparkwheel/)**.

## Quick Reference

| Symbol | Purpose | Sparkwheel Docs |
|--------|---------|-----------------|
| `_target_` | Instantiate a class | [Instantiation](https://project-lighter.github.io/sparkwheel/user-guide/instantiation/) |
| `@path::to::value` | Resolved reference (instantiated object) | [References](https://project-lighter.github.io/sparkwheel/user-guide/references/) |
| `%path::to::value` | Raw reference (unprocessed YAML) | [References](https://project-lighter.github.io/sparkwheel/user-guide/references/) |
| `$expression` | Evaluate Python expression | [Expressions](https://project-lighter.github.io/sparkwheel/user-guide/expressions/) |
| `::` | Path notation (navigate config) | [Basics](https://project-lighter.github.io/sparkwheel/user-guide/basics/) |
| `.` | Access Python attributes | [Expressions](https://project-lighter.github.io/sparkwheel/user-guide/expressions/) |
| `=key:` | Replace operator (override merge) | [Operators](https://project-lighter.github.io/sparkwheel/user-guide/operators/) |
| `~key:` | Delete operator | [Operators](https://project-lighter.github.io/sparkwheel/user-guide/operators/) |

## Lighter Configuration Structure

Every Lighter config has two mandatory sections:

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10

system:
  _target_: lighter.System
  model: ...
  criterion: ...
  optimizer: ...
  dataloaders: ...
```

Optional sections:
```yaml
_requires_:        # Import Python modules
project: ./path    # Custom module directory
vars: ...          # Variables for reuse
args: ...          # Stage-specific arguments (fit, test, etc.)
```

## Essential Syntax

### `_target_`: Instantiate Classes

```yaml
model:
  _target_: torchvision.models.resnet18
  num_classes: 10
```

**Equivalent to:** `torchvision.models.resnet18(num_classes=10)`

[Learn more →](https://project-lighter.github.io/sparkwheel/user-guide/instantiation/)

### `@` and `%`: References

| Type | Syntax | Use Case |
|------|--------|----------|
| **Resolved** (`@`) | `@system::optimizer` | Pass actual object instances |
| **Raw** (`%`) | `%system::metrics::train` | Reuse config to create new instances |

**Example:**
```yaml
scheduler:
  _target_: torch.optim.lr_scheduler.StepLR
  optimizer: "@system::optimizer"  # Resolved: actual optimizer object

metrics:
  train:
    - _target_: torchmetrics.Accuracy
      task: multiclass
      num_classes: 10
  val: "%system::metrics::train"  # Raw: creates new instance
```

[Learn more →](https://project-lighter.github.io/sparkwheel/user-guide/references/)

### `$`: Expressions

Evaluate Python in configs:

```yaml
optimizer:
  _target_: torch.optim.Adam
  params: "$@system::model.parameters()"  # Call model.parameters()
  lr: "$0.001 * 2"  # Result: 0.002
```

[Learn more →](https://project-lighter.github.io/sparkwheel/user-guide/expressions/)

### `::`: Path Notation

Navigate nested configs:

```yaml
@system::model                    # Access model
@system::optimizer::lr            # Access nested value
%::train::batch_size              # Relative reference (sibling)
```

[Learn more →](https://project-lighter.github.io/sparkwheel/user-guide/basics/)

## CLI Overrides

Override any config value from command line:

```bash
# Simple override
lighter fit config.yaml trainer::max_epochs=100

# Nested values
lighter fit config.yaml system::optimizer::lr=0.001

# Multiple overrides
lighter fit config.yaml \
  trainer::max_epochs=100 \
  system::optimizer::lr=0.001 \
  trainer::devices=4
```

[Learn more →](https://project-lighter.github.io/sparkwheel/user-guide/cli/)

## Merging Configs

Combine multiple YAML files for modular experiments:

```bash
# Merge base + experiment
lighter fit base.yaml,experiment.yaml

# Compose from modules
lighter fit base.yaml,models/resnet.yaml,data/cifar10.yaml
```

### Default Merging Behavior

**Dictionaries merge recursively:**
```yaml
# base.yaml
trainer:
  max_epochs: 10
  devices: 1

# experiment.yaml
trainer:
  max_epochs: 100  # Overrides
  accelerator: gpu  # Adds

# Result: max_epochs=100, devices=1, accelerator=gpu
```

**Lists extend (append):**
```yaml
# base.yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint

# experiment.yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.EarlyStopping

# Result: Both callbacks present
```

### Override Merging: `=` and `~`

**Replace with `=`:**
```yaml
# experiment.yaml
trainer:
  =callbacks:  # Replace instead of extend
    - _target_: pytorch_lightning.callbacks.RichProgressBar
```

**Delete with `~`:**
```yaml
# Delete entire key
trainer:
  ~callbacks: null

# Delete list items
trainer:
  ~callbacks: [1, 3]  # Delete indices 1 and 3

# Delete dict keys
system:
  ~dataloaders: ["train", "test"]
```

[Complete merging reference →](https://project-lighter.github.io/sparkwheel/user-guide/operators/)

## Common Lighter Patterns

### Pattern 1: Model → Optimizer

```yaml
system:
  model:
    _target_: torchvision.models.resnet18
    num_classes: 10

  optimizer:
    _target_: torch.optim.Adam
    params: "$@system::model.parameters()"
    lr: 0.001
```

### Pattern 2: Optimizer → Scheduler

```yaml
system:
  optimizer:
    _target_: torch.optim.Adam
    params: "$@system::model.parameters()"
    lr: 0.001

  scheduler:
    _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
    optimizer: "@system::optimizer"
    factor: 0.5
```

### Pattern 3: Reusing Configurations

```yaml
system:
  metrics:
    train:
      - _target_: torchmetrics.Accuracy
        task: multiclass
        num_classes: 10
    val: "%system::metrics::train"  # Reuse

  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      batch_size: 128
      num_workers: 4
    val:
      _target_: torch.utils.data.DataLoader
      batch_size: "%::train::batch_size"  # Relative reference
      num_workers: "%::train::num_workers"
```

### Pattern 4: Variables for Reuse

```yaml
vars:
  batch_size: 32
  num_classes: 10
  base_lr: 0.001

system:
  model:
    _target_: torchvision.models.resnet18
    num_classes: "%vars::num_classes"

  optimizer:
    lr: "%vars::base_lr"

  dataloaders:
    train:
      batch_size: "%vars::batch_size"
```

### Pattern 5: Stage-Specific Arguments

```yaml
args:
  fit:
    ckpt_path: null  # Start from scratch
  test:
    ckpt_path: "checkpoints/best.ckpt"
  predict:
    ckpt_path: "checkpoints/best.ckpt"
    return_predictions: true
```

**Override from CLI:**
```bash
lighter test config.yaml args::test::ckpt_path="other.ckpt"
```

## Common Pitfalls

### 1. Resolved vs Raw Reference

```yaml
# ❌ Wrong: Shares same instance
metrics:
  val: "@system::metrics::train"

# ✅ Correct: Creates new instance
metrics:
  val: "%system::metrics::train"
```

### 2. Path Notation vs Python Attributes

```yaml
# ❌ Wrong: :: is for config paths
params: "$@system::model::parameters()"

# ✅ Correct: . is for Python attributes
params: "$@system::model.parameters()"
```

### 3. Missing $ for Expressions

```yaml
# ❌ Wrong: Treated as string
batch_size: "@vars::base_batch * 2"

# ✅ Correct: Evaluated
batch_size: "$%vars::base_batch * 2"
```

## Advanced Features

Refer to **[Sparkwheel documentation](https://project-lighter.github.io/sparkwheel/)** for advanced usage.

## Next Steps

- [Running Experiments](run.md) - Execute training, testing, prediction
- [Configuration Recipes](recipes.md) - Ready-to-use patterns
- [Troubleshooting](troubleshooting.md) - Debug config errors
- [Sparkwheel Documentation](https://project-lighter.github.io/sparkwheel/) - Complete reference
