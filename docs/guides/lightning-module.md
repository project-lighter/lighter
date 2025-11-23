---
title: Using LightningModule
---

# Using LightningModule

Use your existing PyTorch Lightning code with Lighter's configuration system.

**Key insight**: You don't rewrite your LightningModule. You just add YAML configs.

## When to Use This Approach

**Use LightningModule when:**

- You have existing Lightning code
- You need custom training logic
- You want full control over step methods
- You're integrating with existing projects

**You get:**

- YAML configuration for hyperparameters
- CLI overrides without code changes
- Experiment tracking and versioning
- All PyTorch Lightning features

**You write:**

- All step methods (`training_step`, `validation_step`, etc.)
- `configure_optimizers()`
- Your own logging
- Custom hooks and callbacks

## Basic Example

### Your Existing Module

`model.py`:

```python
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class ImageClassifier(pl.LightningModule):
    """Standard PyTorch Lightning module."""

    def __init__(self, num_classes=10, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 8 * 8, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
```

### Add Config

`config.yaml`:

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10
  accelerator: auto

model:
  _target_: model.ImageClassifier  # Your module!
  num_classes: 10
  learning_rate: 0.001

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

  val_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 32
    dataset:
      _target_: torchvision.datasets.CIFAR10
      root: ./data
      train: false
      transform:
        _target_: torchvision.transforms.ToTensor
```

### Run

```bash
# Run with default config
lighter fit config.yaml

# Override learning rate
lighter fit config.yaml model::learning_rate=0.01

# Override multiple values
lighter fit config.yaml \
  model::learning_rate=0.01 \
  trainer::max_epochs=100 \
  data::train_dataloader::batch_size=64
```

**That's it!** Your existing Lightning code works unchanged.

## Advanced Examples

### Example 1: Complex Network Architecture

Pass complex architectures through config:

`models.py`:

```python
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class FlexibleClassifier(pl.LightningModule):
    """Accepts any network architecture."""

    def __init__(self, network, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters(ignore=['network'])
        self.network = network
        self.lr = learning_rate

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

`config.yaml`:

```yaml
model:
  _target_: models.FlexibleClassifier
  learning_rate: 0.001

  network:
    _target_: torchvision.models.resnet18
    num_classes: 10
    weights: null  # Train from scratch
```

Now you can swap architectures by changing just the config:

```yaml
# Try ResNet50
network:
  _target_: torchvision.models.resnet50
  num_classes: 10

# Try EfficientNet
network:
  _target_: torchvision.models.efficientnet_b0
  num_classes: 10

# Try your custom network
network:
  _target_: my_project.networks.CustomNet
  num_classes: 10
  hidden_dim: 512
```

### Example 2: Multiple Optimizers

For GANs or other multi-optimizer setups:

`gan.py`:

```python
import pytorch_lightning as pl
import torch

class GAN(pl.LightningModule):
    def __init__(self, generator, discriminator, lr_g=0.0002, lr_d=0.0002):
        super().__init__()
        self.save_hyperparameters(ignore=['generator', 'discriminator'])
        self.generator = generator
        self.discriminator = discriminator

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch

        # Train generator
        if optimizer_idx == 0:
            z = torch.randn(real_imgs.size(0), self.generator.latent_dim)
            fake_imgs = self.generator(z)
            g_loss = -torch.mean(self.discriminator(fake_imgs))
            self.log("train/g_loss", g_loss)
            return g_loss

        # Train discriminator
        if optimizer_idx == 1:
            z = torch.randn(real_imgs.size(0), self.generator.latent_dim)
            fake_imgs = self.generator(z).detach()

            d_loss_real = -torch.mean(self.discriminator(real_imgs))
            d_loss_fake = torch.mean(self.discriminator(fake_imgs))
            d_loss = d_loss_real + d_loss_fake

            self.log("train/d_loss", d_loss)
            return d_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr_g,
            betas=(0.5, 0.999)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr_d,
            betas=(0.5, 0.999)
        )
        return [opt_g, opt_d]
```

`config.yaml`:

```yaml
model:
  _target_: gan.GAN
  lr_g: 0.0002
  lr_d: 0.0002

  generator:
    _target_: gan.Generator
    latent_dim: 100
    img_shape: [3, 32, 32]

  discriminator:
    _target_: gan.Discriminator
    img_shape: [3, 32, 32]
```

### Example 3: Custom Metrics

Use torchmetrics or your own:

`models.py`:

```python
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

class MetricsModule(pl.LightningModule):
    def __init__(self, network, num_classes=10, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters(ignore=['network'])
        self.network = network

        # Initialize metrics
        self.train_acc = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=num_classes
        )
        self.val_f1 = torchmetrics.F1Score(
            task='multiclass',
            num_classes=num_classes
        )

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Update metrics
        self.train_acc(logits, y)

        # Log
        self.log("train/loss", loss)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Update metrics
        self.val_acc(logits, y)
        self.val_f1(logits, y)

        # Log
        self.log("val/loss", loss)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
        )
```

### Example 4: Learning Rate Schedulers

Add schedulers in `configure_optimizers`:

```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=self.trainer.max_epochs,
        eta_min=1e-6
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
    }
```

Or use ReduceLROnPlateau:

```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val/loss",  # Metric to monitor
            "interval": "epoch",
        }
    }
```

### Example 5: Gradient Clipping

Add in config or code:

**Option 1: In Config**

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  gradient_clip_val: 0.5
  gradient_clip_algorithm: norm
```

**Option 2: In Code**

```python
def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
    self.clip_gradients(
        optimizer,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm
    )
```

### Example 6: Model Hooks

Use Lightning hooks for custom behavior:

```python
class MyModule(pl.LightningModule):
    def __init__(self, network, learning_rate=0.001):
        super().__init__()
        self.network = network
        self.lr = learning_rate

    def on_train_start(self):
        """Called when training starts."""
        print(f"Starting training with LR: {self.lr}")

    def on_train_epoch_end(self):
        """Called at the end of each epoch."""
        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("train/lr", current_lr)

    def on_validation_epoch_end(self):
        """Called after validation epoch."""
        # Custom validation logic
        pass

    def on_save_checkpoint(self, checkpoint):
        """Modify what gets saved."""
        checkpoint['custom_data'] = {'my_value': 42}

    def on_load_checkpoint(self, checkpoint):
        """Load custom data."""
        custom_data = checkpoint.get('custom_data', {})
        print(f"Loaded custom data: {custom_data}")
```

All Lightning hooks work normally!

## Passing Data from Config

You can pass **any** data structure through config:

### Lists

```yaml
model:
  _target_: models.MyModule
  layer_sizes: [64, 128, 256, 512]
```

```python
def __init__(self, layer_sizes):
    layers = []
    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.ReLU())
    self.network = nn.Sequential(*layers)
```

### Dicts

```yaml
model:
  _target_: models.MyModule
  config:
    hidden_dim: 256
    num_layers: 4
    dropout: 0.1
```

```python
def __init__(self, config):
    self.hidden_dim = config['hidden_dim']
    self.num_layers = config['num_layers']
    self.dropout = config['dropout']
```

### Nested Objects

```yaml
model:
  _target_: models.MyModule
  encoder:
    _target_: models.Encoder
    hidden_dim: 256
  decoder:
    _target_: models.Decoder
    hidden_dim: 256
```

```python
def __init__(self, encoder, decoder):
    self.encoder = encoder
    self.decoder = decoder
```

## Integration with Callbacks

Use any PyTorch Lightning callback:

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 100
  callbacks:
    # Model checkpointing
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      monitor: val/acc
      mode: max
      save_top_k: 3
      filename: 'epoch{epoch:02d}-acc{val/acc:.4f}'

    # Early stopping
    - _target_: pytorch_lightning.callbacks.EarlyStopping
      monitor: val/loss
      patience: 10
      mode: min

    # Learning rate monitor
    - _target_: pytorch_lightning.callbacks.LearningRateMonitor
      logging_interval: epoch

    # Custom callback
    - _target_: my_project.callbacks.MyCustomCallback
      some_param: 42
```

## Integration with Loggers

Use any PyTorch Lightning logger:

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  logger:
    _target_: pytorch_lightning.loggers.TensorBoardLogger
    save_dir: ./logs
    name: my_experiment
```

Or multiple loggers:

```yaml
trainer:
  logger:
    - _target_: pytorch_lightning.loggers.TensorBoardLogger
      save_dir: ./logs

    - _target_: pytorch_lightning.loggers.CSVLogger
      save_dir: ./logs

    - _target_: pytorch_lightning.loggers.WandbLogger
      project: my_project
      name: experiment_1
```

## Testing and Validation

Your module works with all Lightning testing features:

### Fast Dev Run

```bash
lighter fit config.yaml trainer::fast_dev_run=true
```

Runs 1 batch of train/val to catch bugs.

### Validation Only

```bash
lighter validate config.yaml args::validate::ckpt_path=checkpoints/best.ckpt
```

### Testing

```bash
lighter test config.yaml args::test::ckpt_path=checkpoints/best.ckpt
```

### Overfit on Small Batch

```bash
lighter fit config.yaml trainer::overfit_batches=10
```

## Common Patterns

### Pattern 1: Save Hyperparameters

Always save hyperparameters for reproducibility:

```python
def __init__(self, network, learning_rate=0.001, weight_decay=0.0):
    super().__init__()
    # Save all args except network (it's not serializable)
    self.save_hyperparameters(ignore=['network'])
    self.network = network
```

Now `self.hparams` contains your config:

```python
def configure_optimizers(self):
    return torch.optim.Adam(
        self.parameters(),
        lr=self.hparams.learning_rate,
        weight_decay=self.hparams.weight_decay
    )
```

### Pattern 2: Separate Forward from Loss

Keep forward pass separate from loss calculation:

```python
def forward(self, x):
    """Just the forward pass."""
    return self.network(x)

def training_step(self, batch, batch_idx):
    """Loss calculation and logging."""
    x, y = batch
    logits = self(x)  # Call forward
    loss = F.cross_entropy(logits, y)
    self.log("train/loss", loss)
    return loss
```

This makes your model usable for inference.

### Pattern 3: Shared Step Logic

Reduce duplication with shared methods:

```python
def _shared_step(self, batch, stage):
    x, y = batch
    logits = self(x)
    loss = F.cross_entropy(logits, y)
    acc = (logits.argmax(1) == y).float().mean()

    self.log(f"{stage}/loss", loss)
    self.log(f"{stage}/acc", acc)

    return loss

def training_step(self, batch, batch_idx):
    return self._shared_step(batch, "train")

def validation_step(self, batch, batch_idx):
    return self._shared_step(batch, "val")

def test_step(self, batch, batch_idx):
    return self._shared_step(batch, "test")
```

## Migration from Pure Lightning

Already have a Lightning project? Add Lighter in 3 steps:

### Step 1: Add `__lighter__.py`

```bash
cd my_lightning_project
touch __lighter__.py
```

### Step 2: Create Config

`config.yaml`:

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  # Copy your Trainer args from your Python script

model:
  _target_: my_project.models.MyLightningModule
  # Copy your module's __init__ args

data:
  _target_: my_project.data.MyDataModule
  # Or use LighterDataModule
```

### Step 3: Run

```bash
lighter fit config.yaml
```

Your code works unchanged!

## Comparison: LightningModule vs LighterModule

Both approaches are fully supported. Here's when to use each:

| Feature | LightningModule | LighterModule |
|---------|---------------------|---------------|
| Use existing code | ✅ Yes | ❌ Need to adapt |
| Custom training logic | ✅ Full control | ⚠️ Must fit pattern |
| Auto configure_optimizers | ❌ Manual | ✅ Automatic |
| Automatic logging | ❌ Manual | ✅ Dual logging |
| Learning curve | Easy if you know Lightning | Need to learn LighterModule |
| Boilerplate | More | Less |

**Use your LightningModule when:**

- Migrating existing projects
- Need full control
- Have custom training loops
- Team knows Lightning well

**Use LighterModule when:**

- Starting new projects
- Want less boilerplate
- Standard workflows
- Config-driven everything

Both give you YAML configs and CLI overrides!

## Next Steps

- [LighterModule Guide](lighter-module.md) - Compare with the other approach
- [Training Guide](training.md) - Run experiments, save outputs
- [Best Practices](best-practices.md) - Production patterns

## Quick Reference

```python
# LightningModule works as-is
class MyModule(pl.LightningModule):
    def __init__(self, network, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters(ignore=['network'])
        self.network = network

    def training_step(self, batch, batch_idx):
        # Your logic
        return loss

    def configure_optimizers(self):
        # Your optimizer
        return optimizer
```

```yaml
# Reference it in config
model:
  _target_: my_project.models.MyModule
  learning_rate: 0.001
  network:
    _target_: torchvision.models.resnet18
```

```bash
# Run with overrides
lighter fit config.yaml model::learning_rate=0.01
```
