---
title: Quick Start
---

# Quick Start Guide

Get started with Lighter in just 5 minutes! This guide will help you install Lighter and run your first experiment.

## Installation

Install Lighter via pip:

```bash
pip install lighter
```

## Your First Experiment

Let's train a simple ResNet model on CIFAR-10 in just two steps:

### Step 1: Create a Configuration File

Create a file named `config.yaml`:

```yaml
trainer:
    _target_: pytorch_lightning.Trainer
    max_epochs: 5
    accelerator: auto
    devices: auto

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
        train:
            _target_: torch.utils.data.DataLoader
            batch_size: 32
            shuffle: True
            dataset:
                _target_: torchvision.datasets.CIFAR10
                download: True
                root: .datasets
                train: True
                transform:
                    _target_: torchvision.transforms.Compose
                    transforms:
                        - _target_: torchvision.transforms.ToTensor
                        - _target_: torchvision.transforms.Normalize
                          mean: [0.485, 0.456, 0.406]
                          std: [0.229, 0.224, 0.225]

        val:
            _target_: torch.utils.data.DataLoader
            batch_size: 32
            shuffle: False
            dataset:
                _target_: torchvision.datasets.CIFAR10
                download: True
                root: .datasets
                train: False
                transform:
                    _target_: torchvision.transforms.Compose
                    transforms:
                        - _target_: torchvision.transforms.ToTensor
                        - _target_: torchvision.transforms.Normalize
                          mean: [0.485, 0.456, 0.406]
                          std: [0.229, 0.224, 0.225]
```

### Step 2: Run the Training

Execute the training with a single command:

```bash
lighter fit config.yaml
```

That's it! Lighter will:

- Download the CIFAR-10 dataset
- Initialize your model, optimizer, and dataloader
- Train for 5 epochs
- Display training progress with metrics

## Understanding the Output

During training, you'll see:

- Progress bars showing epoch and batch progress
- Training loss values
- Validation metrics (if configured)
- Checkpoint saves (in `lightning_logs/` by default)

## What Just Happened?

With just a YAML file, Lighter:

1. **Loaded your configuration** and validated it
2. **Instantiated all components** (model, optimizer, datasets, etc.)
3. **Set up the training loop** using PyTorch Lightning
4. **Handled all the boilerplate** code for you

## Next Steps

Now that you've run your first experiment:

- **[Installation Guide](installation.md)**: Learn about different installation options and requirements
- **[Zero to Hero Tutorial](../tutorials/zero_to_hero.md)**: Go from beginner to running experiments in 15 minutes
- **[Image Classification Tutorial](../tutorials/image_classification.md)**: Build a complete project with validation and logging
- **[Examples](https://github.com/project-lighter/lighter/tree/main/examples)**: Explore real-world use cases

## Tips for Success

!!! tip "Configuration Power"
    Everything in Lighter is configuration-driven. You can swap models, datasets, or optimizers by just changing the YAML file!

!!! info "No Code Required"
    For standard experiments, you don't need to write any Python code. Just configure and run!

!!! success "Experiment Tracking"
    Lighter automatically integrates with PyTorch Lightning's logging capabilities. You can easily add [loggers](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) like TensorBoard or Weights & Biases by modifying the trainer section in your config.

## Getting Help

- 💬 Join our [Discord Server](https://discord.gg/zJcnp6KrUp) for any questions
- 🐛 Report issues on [GitHub](https://github.com/project-lighter/lighter/issues)
