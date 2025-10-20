# Migrating from PyTorch Lightning

Quick guide for PyTorch Lightning users transitioning to Lighter.

## Key Difference: Configuration Over Code

Lighter uses YAML configs instead of Python classes for experiment definition.

## Conceptual Mapping

| PyTorch Lightning | Lighter |
|-------------------|---------|
| `LightningModule` | `System` + YAML config |
| `Trainer` | `Trainer` (same, from PL) |
| `training_step()` | Handled by `System` |
| `validation_step()` | Handled by `System` |
| `configure_optimizers()` | Optimizer in YAML |
| Custom callbacks | Same (PL callbacks work) |
| Loggers | Same (PL loggers work) |

## Simple Example

### Before (PyTorch Lightning)
```python
class LitModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=10)
        self.criterion = CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

trainer = Trainer(max_epochs=10)
trainer.fit(model, train_loader)
```

### After (Lighter)
```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10

system:
  _target_: lighter.System

  model:
    _target_: torchvision.models.resnet18
    num_classes: 10

  criterion:
    _target_: torch.nn.CrossEntropyLoss

  optimizer:
    _target_: torch.optim.Adam
    params: "$@system#model.parameters()"
    lr: 0.001

  dataloaders:
    train: ...  # DataLoader config
```

```bash
lighter fit config.yaml
```

**Key insight:** Same Trainer, same training logic, different interface.

## What You Need to Learn

Only 3 things are Lighter-specific:

1. **YAML Configuration Syntax** - [Configuration Guide](../how-to/configure.md)
2. **Adapters** (Lighter's key feature) - [Adapters Guide](../how-to/adapters.md)
3. **Project Modules** (optional) - [Project Module Guide](../how-to/project_module.md)

## What Stays the Same

Everything else is PyTorch Lightning:

- **Trainer arguments** - [PL Trainer docs](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
- **Callbacks** - [PL Callback docs](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html)
- **Loggers** - [PL Logger docs](https://lightning.ai/docs/pytorch/stable/extensions/logging.html)
- **Distributed training** - [PL distributed docs](https://lightning.ai/docs/pytorch/stable/common/trainer.html#devices)
- **Profiling** - [PL profiler docs](https://lightning.ai/docs/pytorch/stable/tuning/profiler.html)

## Common Migration Patterns

### Custom Model
Your `nn.Module` works as-is:
```yaml
system:
  model:
    _target_: my_project.models.MyCustomModel
    arg1: value1
```

### Custom Dataset
Your PyTorch `Dataset` works directly:
```yaml
system:
  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: my_project.datasets.MyDataset
        data_path: /path/to/data
```

### Custom Callbacks
PL callbacks work without modification:
```yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.EarlyStopping
      monitor: val_loss
      patience: 5
    - _target_: my_project.callbacks.MyCustomCallback
      arg: value
```

### Learning Rate Schedulers
```yaml
system:
  scheduler:
    _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
    optimizer: "@system#optimizer"
    factor: 0.5
    patience: 10
```

## When NOT to Migrate

Lighter might not fit if:

- You need highly custom training loops (stick with PL or PyTorch)
- You prefer writing code over configuration
- Your project doesn't run many experimental variations

## Next Steps

1. Start with the [Zero to Hero tutorial](../tutorials/zero_to_hero.md)
2. Try the [Image Classification Tutorial](../tutorials/image_classification.md)
3. Understand [Design Philosophy](../design/philosophy.md)
4. Learn about [Adapters](../how-to/adapters.md) (Lighter's superpower)
