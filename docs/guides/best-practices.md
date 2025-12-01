---
title: Best Practices
---

# Best Practices

Production-ready patterns for Lighter projects.

This guide collects battle-tested patterns for structuring projects, organizing configs, and debugging effectively.

## Project Structure

### Recommended Layout

```
my_project/
├── __lighter__.py              # Marker file
├── __init__.py                 # Package root
├── pyproject.toml              # Dependencies
├── README.md                   # Project docs
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet.py
│   │   └── unet.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   └── transforms.py
│   └── callbacks/
│       ├── __init__.py
│       └── custom.py
│
├── experiments/                # Configs
│   ├── base.yaml              # Shared settings
│   ├── image_classification/
│   │   ├── resnet18.yaml
│   │   └── resnet50.yaml
│   └── segmentation/
│       └── unet.yaml
│
├── tests/                      # Tests
│   ├── test_models.py
│   └── test_data.py
│
└── outputs/                    # Generated (gitignore)
    └── YYYY-MM-DD/
```

### Why This Structure?

- **`__lighter__.py`**: Marks project root for auto-discovery
- **`src/`**: Clear separation of code
- **`experiments/`**: Version-controlled configs
- **`outputs/`**: Generated artifacts (not tracked)
- **`tests/`**: Ensure code quality

## Config Organization

### Base + Experiments Pattern

**`experiments/base.yaml`** - Shared settings:

```yaml
# Base configuration shared across experiments

trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 100
  accelerator: auto
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      monitor: val/acc
      mode: max
      save_top_k: 3

    - _target_: pytorch_lightning.callbacks.LearningRateMonitor
      logging_interval: epoch

data:
  _target_: lighter.LighterDataModule
  train_dataloader:
    _target_: torch.utils.data.DataLoader
    num_workers: 4
    pin_memory: true
    persistent_workers: true

  val_dataloader:
    _target_: torch.utils.data.DataLoader
    num_workers: 4
    pin_memory: true
```

**`experiments/resnet18.yaml`** - Specific experiment:

```yaml
# Experiment-specific settings (merged with base.yaml via CLI)
model:
  _target_: src.models.ImageClassifier
  learning_rate: 0.001
  network:
    _target_: torchvision.models.resnet18
    num_classes: 10

data:
  train_dataloader:
    batch_size: 128
    dataset:
      _target_: torchvision.datasets.CIFAR10
      root: ./data
      train: true
```

Run with both configs - they merge in order:

```bash
lighter fit experiments/base.yaml experiments/resnet18.yaml
```

Or override specific values from CLI:

```bash
lighter fit experiments/base.yaml experiments/resnet18.yaml trainer::max_epochs=50
```

!!! info "How config composition works"
    Each config file (and CLI override) is applied sequentially via Sparkwheel's `.update()` method.
    Dictionaries merge recursively, lists extend by default. Use `=key:` to replace instead of merge,
    or `~key:` to delete. See the [Sparkwheel docs](https://project-lighter.github.io/sparkwheel/) for details.

### Use Variables for Reusability

```yaml
vars:
  # Dataset settings
  num_classes: 10
  img_size: 224

  # Training settings
  batch_size: 32
  base_lr: 0.001
  max_epochs: 100

  # Hardware settings
  num_workers: 4
  devices: 1

trainer:
  max_epochs: "%vars::max_epochs"
  devices: "%vars::devices"

model:
  network:
    num_classes: "%vars::num_classes"
  optimizer:
    lr: "%vars::base_lr"

data:
  train_dataloader:
    batch_size: "%vars::batch_size"
    num_workers: "%vars::num_workers"
```

Override easily:

```bash
lighter fit config.yaml vars::batch_size=64 vars::base_lr=0.01
```

### Separate Data Configs

For large datasets, separate data configs:

**`experiments/data/cifar10.yaml`**:

```yaml
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
```

**`experiments/my_experiment.yaml`**:

```yaml
model:
  # ... model config ...
```

Run with both configs:

```bash
lighter fit experiments/data/cifar10.yaml experiments/my_experiment.yaml
```

## Module Design

### Save Hyperparameters

Always save hyperparameters for reproducibility:

```python
class MyModule(pl.LightningModule):
    def __init__(self, network, learning_rate=0.001, weight_decay=0.0):
        super().__init__()
        # Save all args except objects
        self.save_hyperparameters(ignore=['network'])
        self.network = network
```

Access with `self.hparams.learning_rate`.

### Use Type Hints

```python
from typing import Dict, Any
import torch

class MyModule(LighterModule):
    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x, y = batch
        loss = self.criterion(self(x), y)
        return {"loss": loss}
```

### Document __init__ Parameters

Config values map to `__init__` args:

```python
class MyDataset(Dataset):
    """Custom dataset for my task.

    Args:
        root: Path to data directory
        split: One of 'train', 'val', 'test'
        transform: Optional transform to apply
        cache: Whether to cache preprocessed data in memory
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        cache: bool = False
    ):
        ...
```

Users can see available options in docstrings.

### Separate Concerns

**Bad** - Everything in one module:

```python
class MyModule(pl.LightningModule):
    def __init__(self):
        # Network definition
        self.conv1 = nn.Conv2d(...)
        self.conv2 = nn.Conv2d(...)
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        # Metrics
        self.acc = Accuracy()
```

**Good** - Modular design:

```python
class MyModule(pl.LightningModule):
    def __init__(self, network, criterion, metrics):
        super().__init__()
        self.network = network
        self.criterion = criterion
        self.metrics = metrics
```

Config controls composition:

```yaml
model:
  network:
    _target_: src.networks.ResNet
  criterion:
    _target_: torch.nn.CrossEntropyLoss
  metrics:
    _target_: torchmetrics.Accuracy
```

## Data Best Practices

### Use num_workers

```yaml
data:
  train_dataloader:
    num_workers: 4  # Parallelize data loading
    pin_memory: true  # Faster GPU transfer
    persistent_workers: true  # Keep workers alive
```

Start with `num_workers = num_cpus / num_gpus`.

### Prefetch Factor

For slow data loading:

```yaml
data:
  train_dataloader:
    num_workers: 4
    prefetch_factor: 2  # Each worker prefetches 2 batches
```

### Cache Small Datasets

For datasets that fit in RAM:

```python
class CachedDataset(Dataset):
    def __init__(self, root, transform=None, cache=True):
        self.transform = transform
        self.cache = cache

        # Load all data at init
        if self.cache:
            self.data = [self._load_item(i) for i in range(len(files))]

    def __getitem__(self, idx):
        if self.cache:
            item = self.data[idx]
        else:
            item = self._load_item(idx)

        if self.transform:
            item = self.transform(item)

        return item
```

### Validate Data in __init__

Check data availability early:

```python
class MyDataset(Dataset):
    def __init__(self, root, split='train'):
        self.root = Path(root)

        # Validate
        if not self.root.exists():
            raise ValueError(f"Data directory not found: {root}")

        split_file = self.root / f"{split}.txt"
        if not split_file.exists():
            raise ValueError(f"Split file not found: {split_file}")

        # Load
        self.samples = self._load_split(split_file)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split '{split}'")
```

Fails fast with clear error messages.

## Training Best Practices

### Learning Rate Warmup

Stabilize early training:

```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # Warmup for first 1000 steps
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=1000
    )

    # Then cosine decay
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=self.trainer.max_epochs - 10
    )

    # Combine
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[10]
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
        }
    }
```

### Gradient Clipping

Prevent exploding gradients:

```yaml
trainer:
  gradient_clip_val: 1.0
  gradient_clip_algorithm: norm
```

Essential for RNNs and transformers.

### Model EMA (Exponential Moving Average)

Smoother model weights:

```yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.EMA
      decay: 0.999
```

### Mixed Precision

Faster training on modern GPUs:

```yaml
trainer:
  precision: 16  # or "bf16-mixed" for BFloat16
```

### Accumulate Gradients

Simulate large batch size:

```yaml
trainer:
  accumulate_grad_batches: 4  # Effective batch = batch_size × 4
```

## Logging Best Practices

### Log Hyperparameters

Log all hyperparameters for comparison:

```python
def __init__(self, ...):
    super().__init__()
    self.save_hyperparameters()  # Logs to tensorboard/wandb
```

### Dual Logging (Step + Epoch)

LighterModule does this automatically. For Lightning:

```python
def training_step(self, batch, batch_idx):
    loss = ...

    # Log both step and epoch
    self.log("train/loss", loss, on_step=True, on_epoch=True)

    return loss
```

### Log Learning Rate

```yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.LearningRateMonitor
      logging_interval: step
```

### Log Sample Images

```python
def validation_step(self, batch, batch_idx):
    if batch_idx == 0:
        x, y = batch
        pred = self(x)

        # Log first 8 images
        self.logger.experiment.add_images(
            "val/predictions",
            x[:8],
            self.global_step
        )
```

### Use Structured Logging

Group related metrics:

```python
# Good - grouped
self.log("train/loss", loss)
self.log("train/acc", acc)
self.log("val/loss", val_loss)
self.log("val/acc", val_acc)

# Bad - flat
self.log("loss", loss)
self.log("acc", acc)
```

## Debugging Strategies

### Start Simple

1. **Overfit 1 batch**:
   ```bash
   lighter fit config.yaml trainer::overfit_batches=1
   ```
   Should reach ~0 loss quickly.

2. **Fast dev run**:
   ```bash
   lighter fit config.yaml trainer::fast_dev_run=true
   ```
   Catches basic errors.

3. **Limit batches**:
   ```bash
   lighter fit config.yaml trainer::limit_train_batches=10
   ```
   Faster iteration during development.

### Add Assertions

```python
def training_step(self, batch, batch_idx):
    x, y = batch

    # Validate shapes
    assert x.dim() == 4, f"Expected 4D input, got {x.dim()}D"
    assert y.dim() == 1, f"Expected 1D targets, got {y.dim()}D"

    pred = self(x)

    # Validate output
    assert pred.shape[0] == x.shape[0], "Batch size mismatch"
    assert not torch.isnan(pred).any(), "NaN in predictions"

    loss = self.criterion(pred, y)
    return loss
```

Remove in production.

### Log Distributions

```python
def training_step(self, batch, batch_idx):
    loss = ...

    # Log gradient norms
    if batch_idx % 100 == 0:
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.logger.experiment.add_histogram(
                    f"gradients/{name}",
                    param.grad,
                    self.global_step
                )

    return loss
```

### Use Anomaly Detection

For NaN debugging:

```python
torch.autograd.set_detect_anomaly(True)
```

Or in config:

```yaml
trainer:
  detect_anomaly: true
```

Slower, but helps find NaN sources.

## Config Best Practices

### Use Comments

```yaml
model:
  optimizer:
    _target_: torch.optim.AdamW
    params: "$@model::network.parameters()"
    lr: 0.001  # Tuned via LR finder
    weight_decay: 0.01  # L2 regularization

trainer:
  max_epochs: 100  # Convergence around epoch 80
  devices: 4  # 4× A100 GPUs
```

### Avoid Hardcoded Paths

**Bad**:

```yaml
data:
  dataset:
    root: /home/user/data/cifar10  # Breaks on other machines
```

**Good**:

```yaml
vars:
  data_root: ./data  # Relative path

data:
  dataset:
    root: "%vars::data_root/cifar10"
```

Or use environment variables:

```yaml
vars:
  data_root: "$os.environ.get('DATA_ROOT', './data')"
```

### Version Configs

```yaml
# config.yaml
_meta_:
  version: "1.2.0"
  description: "ResNet50 with strong augmentation"
  created: "2024-01-15"
  author: "your-name"

# ... rest of config ...
```

### Keep Configs DRY

Use references to avoid duplication:

```yaml
vars:
  num_classes: 10

model:
  network:
    num_classes: "%vars::num_classes"

  train_metrics:
    - _target_: torchmetrics.Accuracy
      num_classes: "%vars::num_classes"  # Same value
```

## Reproducibility

### Set Seeds

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  deterministic: true  # Reproducible

# In __lighter__.py
import pytorch_lightning as pl
pl.seed_everything(42, workers=True)
```

### Log Everything

```python
def __init__(self, ...):
    super().__init__()
    self.save_hyperparameters()  # Save all args

# In trainer
trainer:
  logger:
    _target_: pytorch_lightning.loggers.WandbLogger
    log_model: true  # Save model to wandb
```

### Version Dependencies

`pyproject.toml`:

```toml
[project]
dependencies = [
    "torch==2.1.0",
    "pytorch-lightning==2.1.0",
    "lighter>=3.0.0",
]
```

### Save Config with Outputs

Lightning does this automatically:

```
outputs/
└── 2024-01-15/
    └── 10-30-45/
        ├── config.yaml  # Exact config used
        └── checkpoints/
```

## Testing

### Unit Test Models

```python
# tests/test_models.py
import pytest
import torch
from src.models import MyModule

def test_forward_pass():
    model = MyModule(network=...)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)

    assert y.shape == (2, 10)
    assert not torch.isnan(y).any()

def test_training_step():
    model = MyModule(...)
    batch = (torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,)))

    result = model.training_step(batch, 0)

    assert "loss" in result
    assert result["loss"].dim() == 0  # Scalar
```

### Integration Test Configs

```python
# tests/test_configs.py
def test_config_loads():
    """Test that config instantiates correctly."""
    from lighter.engine.runner import load_config

    config = load_config("experiments/resnet18.yaml")

    assert config.trainer is not None
    assert config.model is not None
    assert config.data is not None

def test_fast_dev_run(tmp_path):
    """Test full pipeline on 1 batch."""
    import subprocess

    result = subprocess.run(
        [
            "lighter", "fit",
            "experiments/resnet18.yaml",
            f"trainer::default_root_dir={tmp_path}",
            "trainer::fast_dev_run=true",
        ],
        capture_output=True,
    )

    assert result.returncode == 0
```

## Performance Optimization

### Profile First

```bash
lighter fit config.yaml trainer::profiler=simple
```

Identify bottlenecks before optimizing.

### Data Loading

Most common bottleneck:

1. Increase `num_workers`
2. Add `pin_memory=true`
3. Add `persistent_workers=true`
4. Cache small datasets
5. Preprocess data offline

### Model Optimization

```python
# Compile model (PyTorch 2.0+)
self.network = torch.compile(self.network)

# Use channels_last memory format
self.network = self.network.to(memory_format=torch.channels_last)
```

### Mixed Precision

```yaml
trainer:
  precision: "bf16-mixed"  # BFloat16 on A100/H100
```

### Gradient Checkpointing

For large models:

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Trade compute for memory
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return x
```

## Production Deployment

### Export to TorchScript

```python
# After training
model = MyModule.load_from_checkpoint("best.ckpt")
model.eval()

scripted = torch.jit.script(model)
scripted.save("model.pt")
```

### Export to ONNX

```python
model = MyModule.load_from_checkpoint("best.ckpt")
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"},
    },
)
```

### Serve with TorchServe

See [TorchServe docs](https://pytorch.org/serve/) for deployment.

## Common Pitfalls

### ❌ Using `@` for Metrics

```yaml
# WRONG - Shared instance!
val_metrics: "@model::train_metrics"
```

```yaml
# CORRECT - New instance
val_metrics: "%model::train_metrics"
```

### ❌ Forgetting to Call .eval()

```python
# Inference
model = MyModule.load_from_checkpoint("best.ckpt")
model.eval()  # Important!

with torch.no_grad():
    pred = model(x)
```

### ❌ Not Saving Hyperparameters

```python
def __init__(self, learning_rate=0.001):
    super().__init__()
    self.save_hyperparameters()  # Don't forget!
    self.lr = learning_rate
```

### ❌ Leaking Validation Data

```python
# BAD - Using training mode on validation
def validation_step(self, batch, batch_idx):
    self.train()  # ❌ Wrong!
    ...

# GOOD - Lightning handles this
def validation_step(self, batch, batch_idx):
    # Already in eval mode
    ...
```

### ❌ Ignoring Batch Size in Metrics

```python
# BAD
self.log("train/loss", loss)  # Averages across uneven batches

# GOOD
self.log("train/loss", loss, batch_size=x.size(0))
```

## Checklist

Before training:

- [ ] Seeds set for reproducibility
- [ ] Hyperparameters saved (`save_hyperparameters()`)
- [ ] Metrics use `%` not `@`
- [ ] Data loading optimized (`num_workers`, `pin_memory`)
- [ ] Checkpointing configured
- [ ] Logging configured
- [ ] Config tested with `fast_dev_run=true`

Before production:

- [ ] Model validated on held-out test set
- [ ] Inference tested on sample data
- [ ] Export format chosen (TorchScript/ONNX)
- [ ] Dependencies versioned
- [ ] Config and checkpoints saved
- [ ] Documentation updated

## Next Steps

- [Training Guide](training.md) - Run experiments
- [Examples](../examples/image-classification.md) - Complete working code
- [FAQ](../faq.md) - Common questions

## Quick Reference

```python
# Save hyperparameters
self.save_hyperparameters(ignore=['network'])

# Dual logging
self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=x.size(0))

# Assertions (dev only)
assert not torch.isnan(x).any()

# Profile
torch.autograd.set_detect_anomaly(True)
```

```yaml
# Reproducibility
trainer:
  deterministic: true

# Performance
trainer:
  precision: "bf16-mixed"
  accumulate_grad_batches: 4

data:
  train_dataloader:
    num_workers: 4
    pin_memory: true
    persistent_workers: true
```
