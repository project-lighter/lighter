---
title: Get Started
---

# Get Started with Lighter

This guide takes you from installation to running experiments in 15 minutes, using proper project structure from the start.

## Installation

```bash
pip install lighter
```

## The Core Idea

Traditional PyTorch Lightning requires writing training loops:

```python
class MyModule(LightningModule):
    def __init__(self):
        self.model = Model()
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

trainer = Trainer(max_epochs=10)
trainer.fit(module, train_loader, val_loader)
```

**Lighter replaces this with configuration:**

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10

system:
  _target_: lighter.System
  model:
    _target_: MyModel
  criterion:
    _target_: torch.nn.CrossEntropyLoss
  optimizer:
    _target_: torch.optim.Adam
    params: "$@system::model.parameters()"
    lr: 0.001
  dataloaders:
    train: ...
    val: ...
```

```bash
lighter fit config.yaml
```

## Step 1: Create Your Project

Set up a proper project structure (this will pay off as you add experiments):

```bash
mkdir -p my_experiments/experiments
cd my_experiments
touch __init__.py
```

Your project structure:
```
my_experiments/
├── __init__.py
└── experiments/
    └── (configs will go here)
```

## Step 2: Minimal Example

Create `experiments/minimal.yaml`:

```yaml
project: .  # Import from my_experiments/

trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 3

system:
  _target_: lighter.System

  model:
    _target_: torch.nn.Linear
    in_features: 784  # MNIST: 28x28 flattened
    out_features: 10  # 10 digits

  criterion:
    _target_: torch.nn.CrossEntropyLoss

  optimizer:
    _target_: torch.optim.Adam
    params: "$@system::model.parameters()"
    lr: 0.001

  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      batch_size: 64
      dataset:
        _target_: torchvision.datasets.MNIST
        root: ./data
        train: true
        download: true
        transform:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Lambda
              lambd: "$lambda x: x.view(-1)"  # Flatten to 784
```

Run it:

```bash
lighter fit experiments/minimal.yaml
```

You just trained a neural network using only YAML configuration.

## Step 3: Real Example (CIFAR-10)

Create `experiments/cifar10.yaml`:

```yaml
project: .

trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10
  accelerator: auto

system:
  _target_: lighter.System

  model:
    _target_: torchvision.models.resnet18
    num_classes: 10

  criterion:
    _target_: torch.nn.CrossEntropyLoss

  optimizer:
    _target_: torch.optim.Adam
    params: "$@system::model.parameters()"
    lr: 0.001

  metrics:
    train:
      - _target_: torchmetrics.Accuracy
        task: multiclass
        num_classes: 10
    val: "%system::metrics::train"  # Reuse train metrics

  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      batch_size: 128
      shuffle: true
      num_workers: 4
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: ./data
        train: true
        download: true
        transform:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.RandomHorizontalFlip
            - _target_: torchvision.transforms.RandomCrop
              size: 32
              padding: 4
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.4914, 0.4822, 0.4465]
              std: [0.2470, 0.2435, 0.2616]

    val:
      _target_: torch.utils.data.DataLoader
      batch_size: 256
      num_workers: 4
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: ./data
        train: false
        download: true
        transform:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.4914, 0.4822, 0.4465]
              std: [0.2470, 0.2435, 0.2616]
```

Run it:

```bash
lighter fit experiments/cifar10.yaml
```

You now have automatic:
- Training and validation loops
- Metrics computation and logging
- Loss tracking
- Checkpointing

## Step 4: Add Custom Models

Now here's why we set up a proper project structure. Let's add a custom CNN.

Create `models/__init__.py` and `models/simple_cnn.py`:

```bash
mkdir models
touch models/__init__.py
```

```python title="models/simple_cnn.py"
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

Your project now looks like:
```
my_experiments/
├── __init__.py
├── experiments/
│   ├── minimal.yaml
│   └── cifar10.yaml
├── models/
│   ├── __init__.py
│   └── simple_cnn.py
└── data/               # Created by datasets
```

Create `experiments/custom_model.yaml`:

```yaml
project: .  # This imports my_experiments/

trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10

system:
  _target_: lighter.System

  model:
    _target_: models.simple_cnn.SimpleCNN  # Your custom model!
    num_classes: 10

  criterion:
    _target_: torch.nn.CrossEntropyLoss

  optimizer:
    _target_: torch.optim.Adam
    params: "$@system::model.parameters()"
    lr: 0.001

  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      batch_size: 128
      shuffle: true
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: ./data
        train: true
        download: true
        transform:
          _target_: torchvision.transforms.ToTensor
```

Run it:

```bash
lighter fit experiments/custom_model.yaml
```

**This is the key insight**: By setting up proper structure from the start, adding custom components is natural, not a separate concept to learn.

## Understanding the Syntax

Lighter uses **[Sparkwheel](https://project-lighter.github.io/sparkwheel/)** for configuration. Here are the essentials:

### `_target_:` Instantiate a Class

```yaml
model:
  _target_: torch.nn.Linear
  in_features: 784
  out_features: 10
```

**Equivalent to:** `model = torch.nn.Linear(in_features=784, out_features=10)`

Works with any Python class—PyTorch, third-party, or your custom code.

### `project:` Import Custom Modules

```yaml
project: .  # Import from current directory as a Python module
```

This makes `models/`, `datasets/`, `transforms/` etc. importable via `_target_`.

### `$` Evaluate Python Expression

```yaml
optimizer:
  lr: "$0.001 * 2"  # Evaluates to 0.002
```

### `@` Resolved Reference

```yaml
optimizer:
  params: "$@system::model.parameters()"  # Gets actual model instance, calls parameters()
```

Gets the instantiated object (after `_target_` processing).

### `%` Raw Reference

```yaml
metrics:
  train:
    - _target_: torchmetrics.Accuracy
      task: multiclass
      num_classes: 10
  val: "%system::metrics::train"  # Gets raw YAML, creates new instance
```

Gets the unprocessed YAML configuration (before instantiation).

### `::` Path Notation

```yaml
system::model         # Navigate to model definition
system::optimizer::lr # Navigate to nested value
```

Navigate nested config with `::` separator—more concise than `["system"]["model"]`.

!!! tip "Learn More"
    For complete Sparkwheel documentation including advanced features, see **[Sparkwheel docs](https://project-lighter.github.io/sparkwheel/)**.

## CLI Overrides

Change hyperparameters without editing files:

```bash
# Change learning rate
lighter fit experiments/cifar10.yaml system::optimizer::lr=0.01

# Train longer
lighter fit experiments/cifar10.yaml trainer::max_epochs=100

# Use multiple GPUs
lighter fit experiments/cifar10.yaml trainer::devices=2

# Combine multiple overrides
lighter fit experiments/cifar10.yaml \
  trainer::max_epochs=100 \
  system::optimizer::lr=0.001 \
  trainer::devices=4
```

## Organizing Multiple Experiments

As your project grows, organize configs by purpose:

```
my_experiments/
├── __init__.py
├── experiments/
│   ├── baselines/
│   │   ├── resnet18.yaml
│   │   └── resnet50.yaml
│   ├── ablations/
│   │   ├── no_augmentation.yaml
│   │   └── different_optimizer.yaml
│   └── production/
│       └── final_model.yaml
├── models/
│   ├── __init__.py
│   └── simple_cnn.py
└── datasets/           # Add custom datasets here
    ├── __init__.py
    └── my_dataset.py
```

## Merging Configs

Create reusable config components:

```yaml title="experiments/base.yaml"
project: .

trainer:
  _target_: pytorch_lightning.Trainer
  accelerator: auto

system:
  _target_: lighter.System
  criterion:
    _target_: torch.nn.CrossEntropyLoss
```

```yaml title="experiments/resnet18.yaml"
system:
  model:
    _target_: torchvision.models.resnet18
    num_classes: 10
  optimizer:
    _target_: torch.optim.Adam
    params: "$@system::model.parameters()"
    lr: 0.001
```

Combine them:

```bash
lighter fit experiments/base.yaml,experiments/resnet18.yaml
```

Later configs override earlier ones, enabling modular experiment design.

## Testing and Prediction

```bash
# Test trained model
lighter test experiments/cifar10.yaml args::test::ckpt_path=checkpoints/best.ckpt

# Generate predictions
lighter predict experiments/cifar10.yaml args::predict::ckpt_path=checkpoints/best.ckpt
```

## When You Need Adapters

Your dataset returns a dict but your model needs tensors? Use adapters:

```yaml
system:
  adapters:
    train:
      batch:
        _target_: lighter.adapters.BatchAdapter
        input_accessor: "image"  # Extract from dict
        target_accessor: "label"
```

Your loss expects `(target, pred)` instead of `(pred, target)`? Swap them:

```yaml
system:
  adapters:
    train:
      criterion:
        _target_: lighter.adapters.CriterionAdapter
        pred_argument: 1
        target_argument: 0
```

**Adapters make Lighter task-agnostic**—they connect any data format to any model/loss/metric.

[Learn more about adapters →](../how-to/adapters.md)

### Continue Learning

- **[Configuration Guide](../how-to/configuration.md)** - Complete syntax reference
- **[Adapters](../how-to/adapters.md)** - Handle any data format
- **[Recipes](../how-to/recipes.md)** - Ready-to-use patterns
- **[Architecture](../design/overview.md)** - How Lighter works internally

### Quick Reference

```yaml
# Essential Sparkwheel syntax
project: .                          # Import custom modules
_target_: module.ClassName          # Instantiate class
$expression                         # Evaluate Python expression
@path::to::object                   # Resolved reference (instantiated object)
%path::to::config                   # Raw reference (unprocessed YAML)
path::nested::key                   # Path notation (navigate config)
::sibling::key                      # Relative reference (sibling in same section)
=key:                               # Replace operator (override default merge)
~key: null                          # Delete entire key
~key::1: null                       # Delete single list item
~key: [0, 2]                        # Delete multiple items (batch syntax)

# Lighter CLI commands
lighter fit experiments/config.yaml             # Train
lighter validate experiments/config.yaml        # Validate
lighter test experiments/config.yaml            # Test
lighter predict experiments/config.yaml         # Predict

# Override from CLI
lighter fit experiments/cfg.yaml key::path=value

# Merge configs (automatic by default)
lighter fit experiments/base.yaml,experiments/exp.yaml
```

### Project Structure

```
my_experiments/
├── __init__.py                # Make it a module
├── experiments/               # All configs here
│   ├── base.yaml
│   ├── exp1.yaml
│   └── exp2.yaml
├── models/                    # Custom models
│   ├── __init__.py
│   └── my_model.py
├── datasets/                  # Custom datasets
│   ├── __init__.py
│   └── my_dataset.py
└── transforms/                # Custom transforms
    ├── __init__.py
    └── my_transform.py
```

## Getting Help

- **Stuck?** [Troubleshooting Guide](../how-to/troubleshooting.md)
- **Questions?** [FAQ](../faq.md)
- **Coming from Lightning?** [Migration Guide](../migration/from-pytorch-lightning.md)
- **Community?** [Discord](https://discord.gg/zJcnp6KrUp)

## Complete Example

See full examples with this structure in the repository's [projects](https://github.com/project-lighter/lighter/projects/) directory.
