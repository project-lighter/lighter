---
title: Custom Code Guide
---

# Custom Code Guide

Use your own models, datasets, and transforms with Lighter.

Lighter works with **any** Python class. This guide shows how to structure your project and reference your custom code in configs.

## The Project Folder Pattern

Lighter uses a **project folder** with auto-discovery. When you add `__lighter__.py` to your folder, Lighter automatically makes it available as `project`, allowing you to reference your code as `project.module.Class`.

### Quick Example

```
cifar10/
├── __lighter__.py          # Marker file
├── __init__.py             # Python package
├── model.py                # Your models
└── configs/
    └── config.yaml
```

```yaml
# configs/config.yaml
model:
  _target_: project.model.CIFAR10Model  # Auto-discovered!
```

This is the **recommended approach** for organizing Lighter projects.

## Project Structure

A typical Lighter project looks like this:

```
my_project/
├── __lighter__.py          # Marker file (can be empty)
├── __init__.py             # Makes it a package
├── model.py                # Your models
├── data.py                 # Your datasets
├── transforms.py           # Your transforms
├── configs/
│   ├── baseline.yaml
│   └── improved.yaml
└── outputs/                # Created by Lighter
```

### The `__lighter__.py` Marker

This file tells Lighter where your project root is. It can be empty:

```python
# __lighter__.py
# This file can be empty - it just marks your project root
```

Or use it for project-level imports:

```python
# __lighter__.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Any imports here run before config loading
```

### The `__init__.py` Files

Every directory containing code needs `__init__.py`:

```
my_project/
├── __init__.py             # Required
├── models/
│   ├── __init__.py         # Required
│   └── resnet.py
└── data/
    ├── __init__.py         # Required
    └── dataset.py
```

Without `__init__.py`, Python can't import from that directory.

## Using Custom Models

### Example: Custom Model

Create **`model.py`**:

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    """Custom network for CIFAR-10."""

    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### Reference in Config

```yaml
model:
  _target_: lighter.LighterModule

  network:
    _target_: project.model.SimpleNet  # project.file.Class
    num_classes: 10
    dropout: 0.3

  criterion:
    _target_: torch.nn.CrossEntropyLoss

  optimizer:
    _target_: torch.optim.Adam
    params: "$@model::network.parameters()"
    lr: 0.001
```

**Pattern**: `project.module_name.ClassName`

- `project` - auto-discovered from `__lighter__.py`
- `model` - Python file (model.py)
- `SimpleNet` - class name

## Using Custom Datasets

### Example: Custom Dataset

Create **`data.py`**:

```python
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class CustomImageDataset(Dataset):
    """Load images from directory structure."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Assume structure: root_dir/class_name/image.jpg
        self.samples = []
        self.class_to_idx = {}

        for idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if class_dir.is_dir():
                self.class_to_idx[class_dir.name] = idx
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
```

### Reference in Config

```yaml
data:
  _target_: lighter.LighterDataModule

  train_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 32
    shuffle: true
    num_workers: 4
    dataset:
      _target_: project.data.CustomImageDataset
      root_dir: ./data/train
      transform:
        _target_: torchvision.transforms.Compose
        transforms:
          - _target_: torchvision.transforms.Resize
            size: [224, 224]
          - _target_: torchvision.transforms.ToTensor
          - _target_: torchvision.transforms.Normalize
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
```

## Using Custom Transforms

### Example: Custom Transform

Create **`transforms.py`**:

```python
import torch
import random

class RandomCutout:
    """Randomly mask out a square patch from the image."""

    def __init__(self, size=16, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        h, w = img.shape[1:]
        y = random.randint(0, h - self.size)
        x = random.randint(0, w - self.size)

        img[:, y:y+self.size, x:x+self.size] = 0
        return img
```

### Reference in Config

```yaml
data:
  train_dataloader:
    dataset:
      transform:
        _target_: torchvision.transforms.Compose
        transforms:
          - _target_: torchvision.transforms.ToTensor
          - _target_: project.transforms.RandomCutout
            size: 16
            p: 0.5
          - _target_: torchvision.transforms.Normalize
            mean: [0.5, 0.5, 0.5]
            std: [0.5, 0.5, 0.5]
```

## Complete Example: Custom LightningModule

### Step 1: Create Your Module

**`model.py`**:

```python
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class MyCIFAR10Module(pl.LightningModule):
    """Custom training logic for CIFAR-10."""

    def __init__(self, network, learning_rate=0.001, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['network'])
        self.network = network
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Log metrics
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss)
        self.log("train/acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("val/loss", loss)
        self.log("val/acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }
```

### Step 2: Create Config

**`configs/custom.yaml`**:

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 100
  accelerator: auto
  devices: 1
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      monitor: val/acc
      mode: max
      save_top_k: 3

model:
  _target_: project.model.MyCIFAR10Module
  learning_rate: 0.001
  weight_decay: 0.0001
  network:
    _target_: project.model.SimpleNet
    num_classes: 10
    dropout: 0.3

data:
  _target_: lighter.LighterDataModule
  train_dataloader:
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
          - _target_: torchvision.transforms.RandomCrop
            size: 32
            padding: 4
          - _target_: torchvision.transforms.RandomHorizontalFlip
          - _target_: torchvision.transforms.ToTensor
          - _target_: torchvision.transforms.Normalize
            mean: [0.4914, 0.4822, 0.4465]
            std: [0.2470, 0.2435, 0.2616]

  val_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 128
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

### Step 3: Run

```bash
cd my_project
lighter fit configs/custom.yaml
```

## Common Patterns

### Pattern 1: Separate Network and Module

Keep network architecture separate from training logic:

```
my_project/
├── networks/
│   ├── __init__.py
│   ├── resnet.py
│   └── unet.py
├── modules/
│   ├── __init__.py
│   ├── classifier.py
│   └── segmentation.py
└── configs/
    └── config.yaml
```

Config:

```yaml
model:
  _target_: project.modules.classifier.ClassificationModule
  network:
    _target_: project.networks.resnet.ResNet50
    num_classes: 10
```

### Pattern 2: Shared Base Classes

Create base modules for common functionality:

**`modules/base.py`**:

```python
from lighter import LighterModule

class BaseVisionModule(LighterModule):
    """Base module with common vision model utilities."""

    def on_train_start(self):
        # Log model architecture
        self.logger.experiment.add_text(
            "model/architecture",
            str(self.network)
        )

    def log_images(self, images, name, n=8):
        # Helper to log images
        import torchvision
        grid = torchvision.utils.make_grid(images[:n])
        self.logger.experiment.add_image(name, grid, self.global_step)
```

Use in your modules:

```python
from project.modules.base import BaseVisionModule

class MyModule(BaseVisionModule):
    def training_step(self, batch, batch_idx):
        x, y = batch

        # Log images every 100 steps
        if batch_idx % 100 == 0:
            self.log_images(x, "train/inputs")

        # ... rest of training step ...
```

### Pattern 3: Config Inheritance

Use YAML anchors for shared config:

**`configs/base.yaml`**:

```yaml
# Shared settings
defaults: &defaults
  trainer:
    max_epochs: 100
    accelerator: auto

  data:
    train_dataloader:
      batch_size: 32
      num_workers: 4

# Experiment inherits defaults
<<: *defaults

model:
  _target_: project.model.SimpleNet
```

**`configs/large_batch.yaml`**:

```yaml
# Merge with base config
_includes_:
  - base.yaml

# Override just batch size
data:
  train_dataloader:
    batch_size: 128
```

## Organizing Larger Projects

For projects with many modules, organize by functionality:

```
my_project/
├── __lighter__.py
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── classifier.py
│   ├── segmentation.py
│   └── detection.py
├── data/
│   ├── __init__.py
│   ├── datasets.py
│   ├── samplers.py
│   └── augmentation.py
├── utils/
│   ├── __init__.py
│   ├── losses.py
│   └── metrics.py
└── configs/
    ├── classification/
    │   ├── resnet18.yaml
    │   └── efficientnet.yaml
    └── segmentation/
        └── unet.yaml
```

Reference as:

```yaml
model:
  _target_: project.models.classifier.ImageClassifier
  network:
    _target_: project.models.classifier.ResNet18

data:
  train_dataloader:
    dataset:
      _target_: project.data.datasets.CustomDataset
    sampler:
      _target_: project.data.samplers.BalancedSampler

  criterion:
    _target_: project.utils.losses.FocalLoss
```

## Troubleshooting

### Import Error: `ModuleNotFoundError`

**Problem**: `ModuleNotFoundError: No module named 'project'`

**Solution**: Check these in order:

1. `__lighter__.py` exists in project root
2. You're running `lighter` from the directory containing `__lighter__.py`
3. All directories have `__init__.py`

```bash
# Run from here
my_project/
├── __lighter__.py          # ✅ Exists
├── __init__.py             # ✅ Exists
└── model.py

# Not from here
parent/
└── my_project/
    └── ...
```

### Import Error: `cannot import name 'MyClass'`

**Problem**: `ImportError: cannot import name 'MyClass' from 'project.model'`

**Solution**: Check class name matches exactly:

```python
# model.py
class SimpleNet(nn.Module):  # Must match config exactly
    ...
```

```yaml
# config.yaml
network:
  _target_: project.model.SimpleNet  # Exact match
```

### Attribute Error in Config

**Problem**: `AttributeError: 'SimpleNet' object has no attribute 'parameters'`

**Solution**: You're using `::` instead of `.` for Python methods:

```yaml
# ❌ WRONG
params: "$@model::network::parameters()"

# ✅ CORRECT
params: "$@model::network.parameters()"
```

Remember: `::` navigates config, `.` accesses Python attributes.

## Best Practices

### 1. Use Descriptive Module Names

```python
# ❌ Avoid generic names
class Net(nn.Module):
    ...

# ✅ Use descriptive names
class ResNetCIFAR10(nn.Module):
    """ResNet-18 adapted for CIFAR-10."""
    ...
```

### 2. Document __init__ Parameters

Config values map to `__init__` arguments, so document them:

```python
class CustomDataset(Dataset):
    """Custom dataset for my task.

    Args:
        root_dir: Path to data directory
        split: One of 'train', 'val', 'test'
        transform: Optional transform to apply
        target_transform: Optional target transform
    """

    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        ...
```

### 3. Keep Configs DRY with Variables

```yaml
vars:
  num_classes: 10
  img_size: 224
  base_lr: 0.001

model:
  network:
    num_classes: "%vars::num_classes"
  optimizer:
    lr: "%vars::base_lr"

data:
  train_dataloader:
    dataset:
      transform:
        - _target_: torchvision.transforms.Resize
          size: ["%vars::img_size", "%vars::img_size"]
```

### 4. Version Control Configs

```bash
git add configs/baseline.yaml
git commit -m "Add baseline experiment config"
```

Compare experiments:

```bash
git diff configs/baseline.yaml configs/improved.yaml
```

## Complete Project Example

Here's a full working example:

```
cifar10/
├── __lighter__.py
├── __init__.py
├── model.py
├── data.py
├── configs/
│   ├── baseline.yaml
│   └── augmented.yaml
└── README.md
```

**model.py**:

```python
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CIFAR10Module(pl.LightningModule):
    def __init__(self, network, lr=0.001):
        super().__init__()
        self.network = network
        self.lr = lr

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val/loss", loss)
        self.log("val/acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
```

**configs/baseline.yaml**:

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10
  accelerator: auto

model:
  _target_: project.model.CIFAR10Module
  lr: 0.001
  network:
    _target_: project.model.SimpleCNN
    num_classes: 10

data:
  _target_: lighter.LighterDataModule
  train_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 64
    shuffle: true
    dataset:
      _target_: torchvision.datasets.CIFAR10
      root: ./data
      train: true
      download: true
      transform:
        _target_: torchvision.transforms.ToTensor

  val_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 64
    dataset:
      _target_: torchvision.datasets.CIFAR10
      root: ./data
      train: false
      transform:
        _target_: torchvision.transforms.ToTensor
```

**Run it**:

```bash
cd cifar10
lighter fit configs/baseline.yaml
```

## Next Steps

- [Training Guide](training.md) - Run experiments, save outputs
- [Best Practices](best-practices.md) - Production patterns
- [Complete Example](../examples/image-classification.md) - Full CIFAR-10 with all features

## Quick Reference

```python
# Project structure
my_project/
├── __lighter__.py          # Required marker
├── __init__.py             # Required for imports
└── code.py                 # Your code

# Import syntax in config
_target_: project.module.ClassName

# Common issues
# ❌ No __lighter__.py
# ❌ Missing __init__.py
# ❌ Wrong working directory
# ❌ Using :: for Python methods (use . instead)
```
