---
title: First Steps
---

# First Steps with Lighter

Now that you have Lighter installed, let's explore the fundamentals of working with configuration-driven deep learning.

## Understanding the Lighter Philosophy

Lighter follows three core principles:

1. **Configuration over Code**: Define experiments in YAML, not Python
2. **Modularity**: Mix and match components freely
3. **Reproducibility**: Share configs to reproduce experiments exactly

## The Configuration System

### Basic Structure

Every Lighter experiment needs a configuration file with two main sections:

```yaml
trainer:  # PyTorch Lightning Trainer configuration
    _target_: pytorch_lightning.Trainer
    max_epochs: 10

system:  # Lighter System configuration
    _target_: lighter.System
    model: ...
    optimizer: ...
    criterion: ...
    dataloaders: ...
```

### The `_target_` Key

The `_target_` key tells Lighter which Python class to instantiate:

```yaml
model:
    _target_: torch.nn.Linear  # Creates torch.nn.Linear
    in_features: 784
    out_features: 10
```

This is equivalent to:
```python
model = torch.nn.Linear(in_features=784, out_features=10)
```

### Dynamic References

Use `$` syntax to reference other parts of the configuration:

```yaml
optimizer:
    _target_: torch.optim.Adam
    params: "$@system#model.parameters()"  # Reference model parameters
    lr: 0.001
```

## Running Experiments

### Basic Commands

Lighter provides three main commands:

```bash
# Train a model
lighter fit config.yaml

# Validate a model
lighter validate config.yaml --ckpt_path model.ckpt

# Test a model
lighter test config.yaml --ckpt_path model.ckpt

# Make predictions
lighter predict config.yaml --ckpt_path model.ckpt
```

### Command-Line Overrides

Override any configuration value from the command line:

```bash
# Change batch size
lighter fit config.yaml --system.dataloaders.train.batch_size 64

# Change learning rate
lighter fit config.yaml --system.optimizer.lr 0.0001

# Change number of epochs
lighter fit config.yaml --trainer.max_epochs 20
```

## Modifying Your First Experiment

Let's take the quick start config and explore modifications:

### 1. Changing the Model

Replace ResNet with a different architecture:

```yaml
model:
    # Was: torchvision.models.resnet18
    _target_: torchvision.models.efficientnet_b0
    num_classes: 10
```

### 2. Adding Learning Rate Scheduling

```yaml
scheduler:
    _target_: torch.optim.lr_scheduler.StepLR
    optimizer: "$@system#optimizer"
    step_size: 5
    gamma: 0.1
```

### 3. Adding Metrics

```yaml
metrics:
    train:
        accuracy:
            _target_: torchmetrics.Accuracy
            task: multiclass
            num_classes: 10
    val:
        accuracy:
            _target_: torchmetrics.Accuracy
            task: multiclass
            num_classes: 10
        f1:
            _target_: torchmetrics.F1Score
            task: multiclass
            num_classes: 10
```

### 4. Enabling Checkpointing

```yaml
trainer:
    _target_: pytorch_lightning.Trainer
    max_epochs: 10
    callbacks:
        - _target_: pytorch_lightning.callbacks.ModelCheckpoint
          monitor: val/loss
          mode: min
          save_top_k: 3
```

## Organizing Configurations

### Config Inheritance

Split configurations into reusable components:

```yaml
# base_config.yaml
trainer:
    _target_: pytorch_lightning.Trainer
    accelerator: auto
    devices: auto

# experiment.yaml
_base_: base_config.yaml  # Inherit from base

trainer:
    max_epochs: 20  # Override specific values

system:
    model: ...
```

### Modular Configs

Organize complex configs into separate files:

```
configs/
├── trainer/
│   ├── default.yaml
│   └── distributed.yaml
├── model/
│   ├── resnet.yaml
│   └── efficientnet.yaml
├── optimizer/
│   ├── adam.yaml
│   └── sgd.yaml
└── experiment.yaml
```

Then combine them:

```yaml
# experiment.yaml
_base_:
    - trainer/default.yaml
    - model/resnet.yaml
    - optimizer/adam.yaml
```

## Viewing Results

### During Training

Lighter displays real-time metrics:

- Progress bars for epochs and batches
- Loss values
- Custom metrics (if configured)

### After Training

Results are saved in `lightning_logs/`:

```bash
# View with TensorBoard
tensorboard --logdir lightning_logs/

# Check saved checkpoints
ls lightning_logs/version_*/checkpoints/
```

## Common Patterns

### Custom Data Transforms

```yaml
dataset:
    _target_: torchvision.datasets.ImageFolder
    root: ./data
    transform:
        _target_: torchvision.transforms.Compose
        transforms:
            - _target_: torchvision.transforms.Resize
              size: [224, 224]
            - _target_: torchvision.transforms.RandomHorizontalFlip
              p: 0.5
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
```

### Multi-GPU Training

```yaml
trainer:
    _target_: pytorch_lightning.Trainer
    accelerator: gpu
    devices: 2
    strategy: ddp  # Distributed Data Parallel
```

### Mixed Precision Training

```yaml
trainer:
    _target_: pytorch_lightning.Trainer
    precision: 16-mixed  # or bf16-mixed for newer GPUs
```

## Debugging Tips

### 1. Fast Dev Run

Test your configuration quickly:

```yaml
trainer:
    fast_dev_run: 5  # Run only 5 batches
```

### 2. Overfit Batches

Debug by overfitting on a few batches:

```yaml
trainer:
    overfit_batches: 10  # Use only 10 batches
```

### 3. Gradient Checking

Enable gradient anomaly detection:

```yaml
trainer:
    detect_anomaly: true
```

## Best Practices

1. **Start Simple**: Begin with a working config and modify incrementally
2. **Use Version Control**: Track your configs with git
3. **Document Changes**: Comment your YAML files
4. **Validate Early**: Use `fast_dev_run` to catch errors quickly
5. **Modularize**: Break complex configs into reusable components

## Example: Complete Classification Pipeline

Here's a complete example with all components:

```yaml
# Image classification with data augmentation, metrics, and checkpointing
trainer:
    _target_: pytorch_lightning.Trainer
    max_epochs: 30
    accelerator: auto
    devices: auto
    callbacks:
        - _target_: pytorch_lightning.callbacks.ModelCheckpoint
          monitor: val/accuracy
          mode: max
          save_top_k: 3
        - _target_: pytorch_lightning.callbacks.EarlyStopping
          monitor: val/loss
          patience: 5
          mode: min

system:
    _target_: lighter.System

    model:
        _target_: torchvision.models.resnet50
        pretrained: true
        num_classes: 10

    criterion:
        _target_: torch.nn.CrossEntropyLoss

    optimizer:
        _target_: torch.optim.AdamW
        params: "$@system#model.parameters()"
        lr: 0.001
        weight_decay: 0.01

    scheduler:
        _target_: torch.optim.lr_scheduler.CosineAnnealingLR
        optimizer: "$@system#optimizer"
        T_max: 30

    metrics:
        train:
            accuracy:
                _target_: torchmetrics.Accuracy
                task: multiclass
                num_classes: 10
        val:
            accuracy:
                _target_: torchmetrics.Accuracy
                task: multiclass
                num_classes: 10
            top5:
                _target_: torchmetrics.Accuracy
                task: multiclass
                num_classes: 10
                top_k: 5

    dataloaders:
        train:
            _target_: torch.utils.data.DataLoader
            batch_size: 32
            shuffle: true
            num_workers: 4
            dataset:
                _target_: torchvision.datasets.CIFAR10
                root: .datasets
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
                          mean: [0.485, 0.456, 0.406]
                          std: [0.229, 0.224, 0.225]

        val:
            _target_: torch.utils.data.DataLoader
            batch_size: 64
            shuffle: false
            num_workers: 4
            dataset:
                _target_: torchvision.datasets.CIFAR10
                root: .datasets
                train: false
                download: true
                transform:
                    _target_: torchvision.transforms.Compose
                    transforms:
                        - _target_: torchvision.transforms.ToTensor
                        - _target_: torchvision.transforms.Normalize
                          mean: [0.485, 0.456, 0.406]
                          std: [0.229, 0.224, 0.225]
```

## Next Steps

You're now ready to:

- Explore [Tutorials](../tutorials/zero_to_hero.md) for specific use cases
- Learn about [Advanced Configuration](../how-to/configure.md)
- Understand [Adapters](../how-to/adapters.md) for data flow
- Implement [Custom Components](../how-to/project_module.md)

## Getting Help

- 💬 [Discord Community](https://discord.gg/zJcnp6KrUp)
- 📖 [Full Documentation](https://project-lighter.github.io/lighter/)
- 🐛 [GitHub Issues](https://github.com/project-lighter/lighter/issues)
