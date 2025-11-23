---
title: Using LighterModule
---

# Using LighterModule

Build models with less boilerplate using `LighterModule`.

**Key insight**: You write step logic. LighterModule handles optimizers, schedulers, and logging automatically.

## When to Use This Approach

**Use LighterModule when:**

- Starting new projects
- Want less boilerplate code
- Standard training workflows
- Config-driven everything

**You get:**

- Automatic `configure_optimizers()`
- Dual logging (step + epoch)
- Config-driven metrics
- All PyTorch Lightning features

**You write:**

- Step implementations only (`training_step`, `validation_step`, etc.)
- Your model's forward logic
- That's it!

## Basic Example

### Minimal Implementation

`model.py`:

```python
from lighter import LighterModule

class MyModel(LighterModule):
    """Minimal model - just implement steps."""

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)  # Forward pass through self.network
        loss = self.criterion(pred, y)  # Use self.criterion

        # Update metrics
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

### Config

`config.yaml`:

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10
  accelerator: auto

model:
  _target_: model.MyModel

  # Network architecture
  network:
    _target_: torchvision.models.resnet18
    num_classes: 10

  # Loss function
  criterion:
    _target_: torch.nn.CrossEntropyLoss

  # Optimizer (auto-configured!)
  optimizer:
    _target_: torch.optim.Adam
    params: "$@model::network.parameters()"
    lr: 0.001

  # Metrics (optional)
  train_metrics:
    - _target_: torchmetrics.Accuracy
      task: multiclass
      num_classes: 10

  val_metrics: "%model::train_metrics"  # Copy config

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
      transform:
        _target_: torchvision.transforms.ToTensor
```

**That's it!** No `configure_optimizers()`, no manual logging.

## How It Works

LighterModule provides these automatically:

### 1. Forward Pass

```python
def forward(self, x):
    """Calls self.network(x) automatically."""
    return self.network(x)
```

You can override if needed:

```python
def forward(self, x):
    # Custom forward logic
    features = self.network.encoder(x)
    output = self.network.decoder(features)
    return output
```

### 2. Configure Optimizers

```python
def configure_optimizers(self):
    """Auto-creates optimizer and optional scheduler."""
    # Creates optimizer from config
    # Adds scheduler if provided
    # Returns proper format for Lightning
```

No manual implementation needed!

### 3. Automatic Logging

LighterModule logs everything you return:

```python
def training_step(self, batch, batch_idx):
    return {
        "loss": loss,           # Logged as train/loss
        "accuracy": acc,        # Logged as train/accuracy
        "custom_metric": val    # Logged as train/custom_metric
    }
```

**Dual logging**: Logged both per-step and per-epoch automatically.

### 4. Metric Updates

Metrics update automatically if you provide them:

```python
if self.train_metrics:
    self.train_metrics(pred, y)  # Updates all metrics
```

Results logged as:

- `train/Accuracy`
- `train/F1Score`
- etc.

## What LighterModule Provides

### Attributes Available

```python
class MyModel(LighterModule):
    def training_step(self, batch, batch_idx):
        # Available attributes:
        self.network          # From config: model::network
        self.criterion        # From config: model::criterion
        self.optimizer        # From config: model::optimizer
        self.scheduler        # From config: model::scheduler (optional)
        self.train_metrics    # From config: model::train_metrics (optional)
        self.val_metrics      # From config: model::val_metrics (optional)
        self.test_metrics     # From config: model::test_metrics (optional)
```

All optional except `network` (you need something to run!).

### Required Implementations

You **must** implement:

```python
def training_step(self, batch, batch_idx):
    """Required."""
    return {"loss": loss}
```

Optional but common:

```python
def validation_step(self, batch, batch_idx):
    """Optional."""
    return {"loss": loss}

def test_step(self, batch, batch_idx):
    """Optional."""
    return {"loss": loss}

def predict_step(self, batch, batch_idx):
    """Optional."""
    return predictions
```

## Complete Examples

### Example 1: Image Classification

`models.py`:

```python
from lighter import LighterModule

class ImageClassifier(LighterModule):
    """Image classification with metrics."""

    def training_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass
        logits = self(images)

        # Loss
        loss = self.criterion(logits, labels)

        # Metrics
        if self.train_metrics:
            self.train_metrics(logits, labels)

        # Return dict - all values logged automatically
        return {
            "loss": loss,
        }

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        if self.val_metrics:
            self.val_metrics(logits, labels)

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        if self.test_metrics:
            self.test_metrics(logits, labels)

        return {"predictions": logits.argmax(dim=1)}
```

`config.yaml`:

```yaml
model:
  _target_: models.ImageClassifier

  network:
    _target_: torchvision.models.resnet50
    weights: IMAGENET1K_V2  # Pretrained
    num_classes: 10

  criterion:
    _target_: torch.nn.CrossEntropyLoss
    label_smoothing: 0.1

  optimizer:
    _target_: torch.optim.AdamW
    params: "$@model::network.parameters()"
    lr: 0.001
    weight_decay: 0.01

  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    optimizer: "@model::optimizer"
    T_max: 100

  train_metrics:
    - _target_: torchmetrics.Accuracy
      task: multiclass
      num_classes: 10
    - _target_: torchmetrics.F1Score
      task: multiclass
      num_classes: 10
      average: macro

  val_metrics: "%model::train_metrics"
  test_metrics: "%model::train_metrics"
```

### Example 2: Semantic Segmentation

`models.py`:

```python
from lighter import LighterModule
import torch.nn.functional as F

class SemanticSegmentation(LighterModule):
    """Semantic segmentation with dice loss."""

    def training_step(self, batch, batch_idx):
        images, masks = batch

        # Forward
        logits = self(images)

        # Resize logits to match mask size if needed
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        # Loss
        loss = self.criterion(logits, masks)

        # Metrics
        if self.train_metrics:
            preds = logits.argmax(dim=1)
            self.train_metrics(preds, masks)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)

        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        loss = self.criterion(logits, masks)

        if self.val_metrics:
            preds = logits.argmax(dim=1)
            self.val_metrics(preds, masks)

        return {"loss": loss}
```

`config.yaml`:

```yaml
model:
  _target_: models.SemanticSegmentation

  network:
    _target_: segmentation_models_pytorch.Unet
    encoder_name: resnet34
    encoder_weights: imagenet
    in_channels: 3
    classes: 21

  criterion:
    _target_: segmentation_models_pytorch.losses.DiceLoss
    mode: multiclass

  optimizer:
    _target_: torch.optim.Adam
    params: "$@model::network.parameters()"
    lr: 0.0001

  train_metrics:
    - _target_: torchmetrics.JaccardIndex
      task: multiclass
      num_classes: 21

  val_metrics: "%model::train_metrics"
```

### Example 3: Multi-Task Learning

`models.py`:

```python
from lighter import LighterModule

class MultiTaskModel(LighterModule):
    """Multi-task: classification + regression."""

    def __init__(self, network, criterion_cls, criterion_reg,
                 optimizer, alpha=0.5):
        super().__init__(
            network=network,
            criterion=None,  # We have multiple
            optimizer=optimizer
        )
        self.criterion_cls = criterion_cls
        self.criterion_reg = criterion_reg
        self.alpha = alpha  # Task weighting

    def training_step(self, batch, batch_idx):
        images, labels_cls, labels_reg = batch

        # Forward
        out_cls, out_reg = self(images)

        # Two losses
        loss_cls = self.criterion_cls(out_cls, labels_cls)
        loss_reg = self.criterion_reg(out_reg, labels_reg)

        # Combined loss
        loss = self.alpha * loss_cls + (1 - self.alpha) * loss_reg

        # Return all for logging
        return {
            "loss": loss,
            "loss_cls": loss_cls.detach(),
            "loss_reg": loss_reg.detach(),
        }

    def validation_step(self, batch, batch_idx):
        images, labels_cls, labels_reg = batch
        out_cls, out_reg = self(images)

        loss_cls = self.criterion_cls(out_cls, labels_cls)
        loss_reg = self.criterion_reg(out_reg, labels_reg)
        loss = self.alpha * loss_cls + (1 - self.alpha) * loss_reg

        # Accuracy for classification head
        acc = (out_cls.argmax(1) == labels_cls).float().mean()

        return {
            "loss": loss,
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "acc": acc,
        }
```

### Example 4: Custom Forward Pass

Override `forward` for custom logic:

```python
from lighter import LighterModule

class AutoencoderModel(LighterModule):
    """Autoencoder with custom forward."""

    def forward(self, x):
        """Custom forward through encoder-decoder."""
        latent = self.network.encoder(x)
        reconstruction = self.network.decoder(latent)
        return reconstruction, latent

    def training_step(self, batch, batch_idx):
        images, _ = batch

        # Forward returns tuple
        reconstruction, latent = self(images)

        # Reconstruction loss
        loss_recon = self.criterion(reconstruction, images)

        # Optional: regularization on latent
        loss_kl = 0.001 * (latent ** 2).mean()

        loss = loss_recon + loss_kl

        return {
            "loss": loss,
            "loss_recon": loss_recon.detach(),
            "loss_kl": loss_kl.detach(),
        }

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        reconstruction, _ = self(images)
        loss = self.criterion(reconstruction, images)
        return {"loss": loss}
```

## Adding Schedulers

LighterModule handles schedulers automatically:

```yaml
model:
  optimizer:
    _target_: torch.optim.Adam
    params: "$@model::network.parameters()"
    lr: 0.001

  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    optimizer: "@model::optimizer"  # Reference optimizer
    T_max: 100
    eta_min: 0.00001
```

**Supported scheduler types:**

### Step-based

```yaml
scheduler:
  _target_: torch.optim.lr_scheduler.StepLR
  optimizer: "@model::optimizer"
  step_size: 30
  gamma: 0.1
```

### Plateau-based

```yaml
scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  optimizer: "@model::optimizer"
  mode: min
  factor: 0.5
  patience: 10
```

### Warmup

```yaml
scheduler:
  _target_: torch.optim.lr_scheduler.LinearLR
  optimizer: "@model::optimizer"
  start_factor: 0.1
  total_iters: 1000
```

### Chained Schedulers

For complex schedules, override `configure_optimizers`:

```python
def configure_optimizers(self):
    # Warmup then cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        self.optimizer,
        start_factor=0.1,
        total_iters=1000
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer,
        T_max=self.trainer.max_epochs - 10
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        self.optimizer,
        schedulers=[warmup, cosine],
        milestones=[10]
    )

    return {
        "optimizer": self.optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
        }
    }
```

## Working with Metrics

### Single Metric

```yaml
model:
  train_metrics:
    _target_: torchmetrics.Accuracy
    task: multiclass
    num_classes: 10
```

Update in code:

```python
if self.train_metrics:
    self.train_metrics(preds, targets)
```

### Multiple Metrics

```yaml
model:
  train_metrics:
    - _target_: torchmetrics.Accuracy
      task: multiclass
      num_classes: 10
    - _target_: torchmetrics.F1Score
      task: multiclass
      num_classes: 10
    - _target_: torchmetrics.Precision
      task: multiclass
      num_classes: 10
```

Update in code (same!):

```python
if self.train_metrics:
    self.train_metrics(preds, targets)  # Updates all metrics
```

### Metric Collections

Use `MetricCollection` for grouped metrics:

```yaml
model:
  train_metrics:
    _target_: torchmetrics.MetricCollection
    metrics:
      accuracy:
        _target_: torchmetrics.Accuracy
        task: multiclass
        num_classes: 10
      f1:
        _target_: torchmetrics.F1Score
        task: multiclass
        num_classes: 10
```

### Per-Class Metrics

```yaml
model:
  val_metrics:
    - _target_: torchmetrics.Accuracy
      task: multiclass
      num_classes: 10
      average: none  # Per-class accuracy
```

## Custom Initialization

Need custom setup? Override `__init__`:

```python
from lighter import LighterModule

class MyModel(LighterModule):
    def __init__(self, network, criterion, optimizer,
                 special_param=42):
        super().__init__(
            network=network,
            criterion=criterion,
            optimizer=optimizer
        )

        # Custom initialization
        self.special_param = special_param
        self.custom_buffer = []

        # Freeze backbone
        for param in self.network.backbone.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        # Use custom attributes
        if batch_idx % self.special_param == 0:
            self.custom_buffer.append(batch_idx)

        # ... rest of step ...
```

Config:

```yaml
model:
  _target_: models.MyModel
  special_param: 100
  network:
    _target_: ...
  criterion:
    _target_: ...
  optimizer:
    _target_: ...
```

## Using Lightning Hooks

All Lightning hooks work:

```python
class MyModel(LighterModule):
    def on_train_start(self):
        print("Training starting!")

    def on_train_epoch_end(self):
        # Log custom metrics
        avg_loss = self.trainer.callback_metrics.get('train/loss')
        if avg_loss is not None:
            print(f"Epoch {self.current_epoch}: {avg_loss:.4f}")

    def on_validation_epoch_end(self):
        # Custom validation logic
        pass

    def on_save_checkpoint(self, checkpoint):
        # Add custom data
        checkpoint['my_data'] = self.custom_buffer

    def on_load_checkpoint(self, checkpoint):
        # Load custom data
        self.custom_buffer = checkpoint.get('my_data', [])
```

## Differential Learning Rates

Use parameter groups in optimizer config:

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

## Gradient Accumulation

Use Trainer config:

```yaml
trainer:
  accumulate_grad_batches: 4  # Accumulate 4 batches
```

Effective batch size = batch_size × accumulate_grad_batches.

## Mixed Precision Training

```yaml
trainer:
  precision: 16  # Use 16-bit precision
```

Or:

```yaml
trainer:
  precision: "bf16-mixed"  # BFloat16 mixed precision
```

## Saving Predictions

Override `predict_step`:

```python
def predict_step(self, batch, batch_idx):
    images, _ = batch
    predictions = self(images)

    return {
        "predictions": predictions.argmax(dim=1),
        "probabilities": predictions.softmax(dim=1),
    }
```

Run:

```bash
lighter predict config.yaml
```

Use Writers to save to files - see [Training Guide](training.md#saving-predictions).

## Validation Without Training

```python
def validation_step(self, batch, batch_idx):
    # Just metrics, no loss needed
    x, y = batch
    pred = self(x)

    if self.val_metrics:
        self.val_metrics(pred, y)

    return {}  # Empty dict is fine
```

## Common Patterns

### Pattern 1: Return Dict for Auto-Logging

```python
def training_step(self, batch, batch_idx):
    # Everything in the dict gets logged automatically
    return {
        "loss": loss,                    # Required
        "accuracy": accuracy,            # Optional
        "learning_rate": current_lr,     # Optional
        "custom_metric": custom_value,   # Optional
    }
```

Logs as:

- `train/loss`
- `train/accuracy`
- `train/learning_rate`
- `train/custom_metric`

### Pattern 2: Conditional Logging

```python
def training_step(self, batch, batch_idx):
    loss = self.criterion(self(x), y)

    # Only log images every 100 steps
    if batch_idx % 100 == 0:
        self.logger.experiment.add_images(
            "train/images",
            x[:8],
            self.global_step
        )

    return {"loss": loss}
```

### Pattern 3: Custom Metric Update

```python
def validation_step(self, batch, batch_idx):
    x, y = batch
    pred = self(x)
    loss = self.criterion(pred, y)

    # Update specific metrics with transforms
    if self.val_metrics:
        # Apply softmax before metric
        probs = pred.softmax(dim=1)
        self.val_metrics(probs, y)

    return {"loss": loss}
```

## Comparison: LighterModule vs LightningModule

| Feature | LighterModule | LightningModule |
|---------|--------------|---------------------|
| Boilerplate | Less | More |
| configure_optimizers | Automatic | Manual |
| Logging | Automatic dual logging | Manual |
| Metrics | Config-driven | Code-driven |
| Learning curve | Learn LighterModule | Just Lightning |
| Flexibility | Standard patterns | Full control |
| Migration | Adapt existing code | Use as-is |

**Choose LighterModule when:**

- Starting fresh
- Want minimal code
- Standard workflows
- Config everything

**Choose LightningModule when:**

- Have existing code
- Need custom logic
- Want full control
- Complex training loops

Both give you YAML configs and CLI overrides!

## Next Steps

- [Lightning Module Guide](lightning-module.md) - Compare with the other approach
- [Training Guide](training.md) - Run experiments, save outputs
- [Best Practices](best-practices.md) - Production patterns

## Quick Reference

```python
from lighter import LighterModule

class MyModel(LighterModule):
    # Optional custom __init__
    def __init__(self, network, criterion, optimizer, **kwargs):
        super().__init__(
            network=network,
            criterion=criterion,
            optimizer=optimizer
        )

    # Required: training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)

        if self.train_metrics:
            self.train_metrics(pred, y)

        return {"loss": loss}

    # Optional: validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)

        if self.val_metrics:
            self.val_metrics(pred, y)

        return {"loss": loss}
```

```yaml
# Config
model:
  _target_: models.MyModel
  network:
    _target_: ...
  criterion:
    _target_: ...
  optimizer:
    _target_: ...
    params: "$@model::network.parameters()"
```
