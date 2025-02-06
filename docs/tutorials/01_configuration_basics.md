# Configuration Basics

## Introduction

Configuration management is key for reproducible and flexible ML experiments. Lighter uses config files to declaratively define experiments, from hyperparameters to models. This tutorial covers Lighter's configuration system fundamentals for effective experiment customization and management.

## Basic Structure of `config.yaml`

Lighter uses YAML configs to define experiments. A typical `config.yaml` is organized into key sections:

*   **`trainer`**: Configures PL `Trainer` (accelerators, devices, callbacks).
*   **`system`**: Defines core system components (model, criterion, optimizer, scheduler, dataloaders).
*   **`args`**: CLI overrides for config values.

Here's a minimal example illustrating the basic structure:

```yaml title="config.yaml"
trainer:
    _target_: pytorch_lightning.Trainer
    max_epochs: 10

system:
    _target_: lighter.System

    model:
        _target_: torch.nn.Linear
        in_features: 100
        out_features: 10

    criterion:
        _target_: torch.nn.CrossEntropyLoss

    optimizer:
        _target_: torch.optim.Adam
        params: "$@system#model.parameters()"
        lr: 0.001

    dataloaders:
        train:
            _target_: torch.utils.data.DataLoader
            dataset:
                _target_: torch.utils.data.TensorDataset
                tensors:
                    - _target_: torch.randn
                      size: [1000, 100]
                    - _target_: torch.randint
                      low: 0
                      high: 10
                      size: [1000]
            batch_size: 32
```

In this example, we define a simple linear model, a cross-entropy loss, and an Adam optimizer. The `dataloaders` section sets up a basic training dataloader using random tensors.

### Stages in Lighter

Lighter uses "stages" (`fit`, `validate`, `test`, `predict`). When running e.g. `lighter fit`, Lighter prunes config to include only relevant sections for `fit` stage, optimizing resource use and clarity.

### Command Line Arguments with `args`

`args` section in `config.yaml` allows overriding config values from CLI, useful for hyperparameter tuning or quick experiments without config edits. More details later.

## MONAI Bundle Syntax for Class Instantiation

Lighter uses MONAI Bundle config style for concise, flexible class specification in YAML. Key element: `_target_` key.

To instantiate a class, specify its fully qualified name with `_target_` and constructor arguments as key-value pairs.

Example: `torch.optim.Adam` optimizer:

```yaml
optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
    weight_decay: 1.0e-5
```

`_target_: torch.optim.Adam` instantiates `Adam` class from `torch.optim` module. `lr` and `weight_decay` are `Adam` constructor arguments.

Syntax applies to all configurable Lighter components (models, datasets, transforms, etc.).

### Example: Defining a Model

```yaml
system:
  model:
    _target_: torchvision.models.resnet18
    pretrained: false
    num_classes: 10
```

Config snippet defines `resnet18` model from `torchvision.models`, `pretrained=false`, `num_classes=10`.

### Example: Defining a Dataset

```yaml
system:
  dataloaders:
    train:
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: .datasets/
        download: true
        transform:
          _target_: torchvision.transforms.ToTensor
```

Defines CIFAR10 dataset with `ToTensor` transform.

## Validation with Cerberus

Lighter uses [Cerberus](https:// Cerberus.readthedocs.io/en/stable/) for config validation. It ensures `config.yaml` adheres to schema defining expected structure/types.

Lighter auto-validates config files. `ValidationError` is raised for discrepancies, with messages for issue fixing.

### Example Validation Error

Let's say you incorrectly specify `max_epochs` as a string instead of an integer in your `config.yaml`:

```yaml
trainer:
    _target_: pytorch_lightning.Trainer
    max_epochs: "string"  # Incorrect type - should be an integer
```

Running Lighter with this configuration will result in a `ValidationError` similar to:

```
ValidationError: Configuration validation failed.
Path: 'trainer.max_epochs', Error: Must be of integer type
```

These validation errors help quickly identify and fix config issues.

## Overriding Configuration from CLI

Lighter's `args` section in `config.yaml` allows CLI overrides, useful for hyperparameter tuning, ablation studies, and quick adjustments.

`args` section mirrors main config, targeting specific override parameters.

### Basic Overriding Syntax

Override value using dot notation to specify parameter path. Example: override `trainer.max_epochs` to `20`:

```bash
lighter fit config.yaml args.fit.trainer.max_epochs=20
```

`args.fit.trainer.max_epochs=20` appended to `lighter fit config.yaml` command. `fit` specifies override applies to `fit` stage.

### Overriding Nested Parameters

Override nested parameters similarly. Example: change optimizer learning rate:

```bash
lighter fit config.yaml args.fit.system.optimizer.lr=0.01
```

### Overriding Multiple Parameters

Override multiple parameters in one command, separated by spaces:

```bash
lighter fit config.yaml args.fit.trainer.max_epochs=20 args.fit.system.optimizer.lr=0.01
```

### Structure of the `args` Section

`args` section in `config.yaml` should reflect stages and sections for override. Example:

```yaml title="config.yaml"
trainer:
    _target_: pytorch_lightning.Trainer
    max_epochs: 10

system:
    _target_: lighter.System
    optimizer:
        _target_: torch.optim.Adam
        lr: 0.001

args:
    fit: # Overrides for the 'fit' stage
        trainer:
            max_epochs: 20 # Example override for max_epochs
        system:
            optimizer:
                lr: 0.01 # Example override for learning rate
```

`args.fit` section pre-defines potential `fit` stage overrides. Applied only if specified in CLI. Without `args` overrides, values from `trainer` and `system` sections are used (e.g., `max_epochs=10`, `lr=0.001`).

## Advanced Configuration Customization

Lighter config system supports advanced customization: nested configs and references.

### Nested Configurations

Create nested configs for organized `config.yaml`, useful for complex experiments.

Example: nested config:

```yaml
system:
  model:
    _target_: my_project.models.ComplexModel
    encoder:
      _target_: my_project.models.Encoder
      num_layers: 4
      hidden_dim: 256
    decoder:
      _target_: my_project.models.Decoder
      num_layers: 2
      output_dim: 10
```

`model` section has nested `encoder` and `decoder` configs, each defining class/arguments.

### References within the Config File

Lighter allows references in `config.yaml` to avoid repetition and improve maintainability. Reference parameters using syntax `"$@[section]#[parameter_path]"`.

Example from basic structure:

```yaml
optimizer:
    _target_: torch.optim.Adam
    params: "$@system#model.parameters()" # Reference to model parameters
    lr: 0.001
```

`params: "$@system#model.parameters()"` references `model` parameters in `system` section, ensuring optimizer uses correct model parameters.

Use references to link config parts, making it dynamic and less error-prone.

## Recap and Next Steps

Tutorial: learned Lighter's config system basics:

*   Basic `config.yaml` structure, key sections.
*   MONAI Bundle syntax (`_target_` for class instantiation).
*   Cerberus validation, error handling.
*   CLI config overrides (`args` section).
*   Advanced customization (nested configs, references).

Equipped to create/customize Lighter configs for DL experiments.

Next tutorials: [image classification](02_image_classification.md), [semantic segmentation](03_semantic_segmentation.md). See [How-To guides](../how-to/02_debugging_config_errors.md) for debugging, [Design section](../design/02_configuration_system.md) for deeper dive.
