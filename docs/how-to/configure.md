# Configuration Guide

Lighter's YAML configuration system provides a powerful, modular way to define experiments. This guide covers everything from basics to advanced patterns.

## Quick Start Example

```yaml title="config.yaml"
# Minimal working configuration
trainer:
    _target_: pytorch_lightning.Trainer
    max_epochs: 5

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
```

Run with: `lighter fit config.yaml`

## Quick Reference 🚀

### Config Syntax Cheat Sheet

| Symbol | Purpose | Example |
|--------|---------|------|
| `_target_` | Instantiate a class | `_target_: torch.nn.Linear` |
| `%` | Text reference (copy YAML) | `val: "%system#metrics#train"` |
| `@` | Object reference (Python instance) | `optimizer: "@system#optimizer"` |
| `$` | Evaluate Python expression | `lr: "$0.001 * 2"` |
| `#` | Navigate config paths | `@system#model#parameters` |
| `.` | Access object attributes | `$@system#model.parameters()` |

### Common Patterns

```yaml
# Pattern 1: Model with optimizer
model:
    _target_: torchvision.models.resnet18
    pretrained: true
optimizer:
    _target_: torch.optim.Adam
    params: "$@system#model.parameters()"  # Get model parameters
    lr: 0.001

# Pattern 2: Scheduler with optimizer reference
scheduler:
    _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
    optimizer: "@system#optimizer"  # Reference optimizer instance
    factor: 0.5

# Pattern 3: Reusing configurations
metrics:
    train:
        - _target_: torchmetrics.Accuracy
          task: multiclass
          num_classes: 10
    val: "%system#metrics#train"  # Copy train metrics config
```

## Config Structure

### Mandatory Sections

Lighter uses YAML configs to define experiments. A typical config is organized into two mandatory sections:

*   **`trainer`**: Configures [`Trainer` (PyTorch Lightning)](https://lightning.ai/docs/pytorch/stable/common/trainer.html).
*   **`system`**: Defines [`System` (Lighter)](../../reference/system/#lighter.system.System) that encapsulates components such as model, criterion, optimizer, flows, or dataloaders.

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

    flows:
        train:
            _target_: lighter.flow.Flow
            batch: ["input", "target"]
            model: ["input"]
            criterion: ["pred", "target"]
            metrics: ["pred", "target"]

    dataloaders:
        train:
            _target_: torch.utils.data.DataLoader
            batch_size: 32
            shuffle: True
            dataset:
                _target_: torch.utils.data.TensorDataset
                tensors:
                    - _target_: torch.randn
                      size: [1000, 100]
                    - _target_: torch.randint
                      low: 0
                      high: 10
                      size: [1000]
```

In this example, we define a simple linear model, a cross-entropy loss, and an Adam optimizer. The `dataloaders` section sets up a basic training dataloader using random tensors.


### Optional Sections

In addition to mandatory `trainer` and `system` sections, you can include the following optional sections:

*   **`_requires_`**: Evaluated before the rest of the config. Useful for importing modules used by Python expressions in the config as explained in [Evaluating Python Expressions](#evaluating-python-expressions).
*   **`project`**: Path to your project directory. Used to import custom modules. For more details, see [Project Module](../how-to/project_module.md).
*   **`vars`**: Store variables for use in other parts of the config. Useful to avoid repetition and easily update values. See [Referencing Other Components](#referencing-other-components).
*   **`args`**: Arguments to pass to the the stage of the experiment being run.


#### Defining `args`

Lighter operates in *stages*: `fit`, `validate`, `test`, and `predict`. We will cover these in the [Run](run.md) guide in detail.

To pass arguments to a stage, use the `args` section in in your config. For example, to set the `ckpt_path` argument of the `fit` stage/method in your config:

```yaml
args:
    fit:
        ckpt_path: "path/to/checkpoint.ckpt"

# ... Rest of the config
```

or pass/override it from the command line:

```bash
lighter fit experiment.yaml --args#fit#ckpt_path="path/to/checkpoint.ckpt"
```

The equivalent of this in Python would be:

```python
Trainer.fit(model, ckpt_path="path/to/checkpoint.ckpt")
```

where `model` is an instance of `System` defined in the `experiment.yaml`.

## Config Syntax

Lighter relies on [MONAI Bundle configuration system](https://docs.monai.io/en/stable/config_syntax.html) to define and instantiate its components in a clear, modular fashion. This system allows you to separate your code from configuration details by specifying classes, functions, and their initialization parameters in YAML.

### Instantiating a Class

To create an instance of a class, use the `_target_` key with the fully qualified name of the class, and list its constructor arguments as key-value pairs. This approach is used for all configurable Lighter components (models, datasets, transforms, optimizers, etc.).

Example:

```yaml hl_lines="2"
model:
    _target_: torchvision.models.resnet18
    pretrained: False
    num_classes: 10
```

Here, `_target_: torchvision.models.resnet18` directs Lighter to instantiate the `resnet18` model from the `torchvision.models` module using `pretrained` and `num_classes` as constructor arguments. This is equivalent to `torchvision.models.resnet18(pretrained=False, num_classes=10)` in Python.

### Referencing Other Components

Referencing can be achieved either using `%` or `@`.

- `%` textually replaces the reference with the YAML value that it points to.

- `@` replaces the reference with the Python evaluated value that it points to.

To understand the difference, consider the following example:

```yaml hl_lines="4 8"
system:
# ...
    metrics:
        train:
            - _target_: torchmetrics.classification.AUROC
              task: binary
        # Or use relative referencing "%#train" for the same effect
        val: "%system#metrics#train" # (1)!
```

1.  Reference to the same definition as `train`, not the same instance.

In this example, `val: "%system#metrics#train"` creates a new instance of `torchmetrics.classification.AUROC` metric with the same definition as the referenced `train` metric. This is because `%` is a textual reference, and the reference is replaced with the YAML value it points to. If we used `@` instead of `%`, both `train` and `val` would point to the same instance of `AUROC`, which is not the desired behavior.

On the other hand, when defining a scheduler, we want to reference the instantiated optimizer. In this case, we use `@`:
```yaml hl_lines="5"
system:
# ...
    scheduler:
        _target_: torch.optim.lr_scheduler.StepLR
        optimizer: "@system#optimizer" # (1)!
        step_size: 1
```

1.  Reference to the instantiated optimizer.

### Evaluating Python Expressions

Sometimes, you may need to evaluate Python expressions in your config. To indicate that, use `$` before the expression.
For example, we can dinamically define the `min_lr` of a `scheduler` to a fraction of `optimizer#lr`:

```yaml hl_lines="6"
system:
# ...
    scheduler:
        _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
        optimizer: "@system#optimizer"
        end_lr: "$@system#optimizer#lr * 0.1" # (1)!
```

1.  `$` denotes that the expression should be run as Python code.


**Note:** you will regularly use the combination of evaluation and referencing to pass `model.parameters()` to your optimizer:

```yaml hl_lines="3"
optimizer:
    _target_: torch.optim.Adam
    params: "$@system#model.parameters()" # (1)!
    lr: 0.001
```

1. It first fetches the evaluated `"system#model"`, and then runs `.parameters()` on it, as indicated by the `"$"` prefix.


!!! note "`#` vs. `@`"

    - **`#`** — Returns the raw config object (no instantiation). Use this when you want the configuration itself.
    - **`@`** — Instantiate the referenced config definition. Use this when you need the actual runtime object.


### Overriding Config from CLI

Any parameter in the config can be overridden from the command line. Consider the following config:

```yaml title="config.yaml"
trainer:
    max_epochs: 10
    callbacks:
        - _target_: pytorch_lightning.callbacks.EarlyStopping
          monitor: val_acc
        - _target_: pytorch_lightning.callbacks.ModelCheckpoint
          monitor: val_acc
# ...
```

To change `max_epochs` in `trainer` from `10` to `20`:

```bash
lighter fit config.yaml --trainer#max_epochs=20
```

To override an element of a list, simply specify its index:

```bash
lighter fit config.yaml --trainer#callbacks#1#monitor="val_loss"
```

### Merging Configs

You can merge multiple configs by combining them with `,`:

```bash
lighter fit config1.yaml,config2.yaml
```

This will merge the two configs, with the second config overriding any conflicting parameters from the first one.

#### Advanced Merging Patterns

```bash
# Modular configuration
lighter fit base.yaml,models/unet.yaml,data/brats.yaml,train/adam.yaml

# Environment-specific configs
lighter fit config.yaml,envs/local.yaml  # For local development
lighter fit config.yaml,envs/cluster.yaml  # For cluster training
```

## Pro Tips 💡

### 1. Use Variables for DRY Configs
```yaml
vars:
    batch_size: 32
    num_classes: 10
    base_lr: 0.001

system:
    model:
        _target_: torchvision.models.resnet18
        num_classes: "%vars#num_classes"

    optimizer:
        _target_: torch.optim.Adam
        lr: "%vars#base_lr"

    dataloaders:
        train:
            batch_size: "%vars#batch_size"
        val:
            batch_size: "$%vars#batch_size * 2"  # Double for validation
```

### 2. Conditional Configurations
```yaml
# Use Python expressions for conditional logic
system:
    model:
        _target_: "$'torchvision.models.resnet50' if %vars#large_model else 'torchvision.models.resnet18'"
        pretrained: true
```

### 3. Dynamic Imports in _requires_
```yaml
_requires_:
    - "$import math"
    - "$import numpy as np"

vars:
    # Now you can use imported modules
    pi_squared: "$math.pi ** 2"
    random_seed: "$np.random.randint(0, 1000)"
```

## Common Configuration Recipes

### Recipe 1: Multi-GPU Training Setup
```yaml
trainer:
    _target_: pytorch_lightning.Trainer
    devices: -1  # Use all available GPUs
    strategy: ddp  # Distributed Data Parallel
    precision: "16-mixed"  # Mixed precision training

system:
    dataloaders:
        train:
            batch_size: 32  # Per GPU
            num_workers: 4
            pin_memory: true
            persistent_workers: true
```

### Recipe 2: Experiment Tracking
```yaml
trainer:
    logger:
        - _target_: pytorch_lightning.loggers.TensorBoardLogger
          save_dir: logs
          name: experiment_name
          version: "$import datetime; datetime.datetime.now().strftime('%Y%m%d_%H%M%S')"
        - _target_: pytorch_lightning.loggers.WandbLogger
          project: my_project
          name: experiment_name
```

### Recipe 3: Advanced Callbacks
```yaml
trainer:
    callbacks:
        - _target_: pytorch_lightning.callbacks.ModelCheckpoint
          monitor: val_loss
          mode: min
          save_top_k: 3
          filename: "{epoch}-{val_loss:.4f}"

        - _target_: pytorch_lightning.callbacks.EarlyStopping
          monitor: val_loss
          patience: 10
          mode: min

        - _target_: pytorch_lightning.callbacks.LearningRateMonitor
          logging_interval: step
```

### Recipe 4: Complex Data Augmentation
```yaml
system:
    dataloaders:
        train:
            dataset:
                transform:
                    _target_: torchvision.transforms.Compose
                    transforms:
                        - _target_: torchvision.transforms.RandomResizedCrop
                          size: 224
                          scale: [0.8, 1.0]
                        - _target_: torchvision.transforms.RandomHorizontalFlip
                          p: 0.5
                        - _target_: torchvision.transforms.ColorJitter
                          brightness: 0.4
                          contrast: 0.4
                        - _target_: torchvision.transforms.ToTensor
                        - _target_: torchvision.transforms.Normalize
                          mean: [0.485, 0.456, 0.406]
                          std: [0.229, 0.224, 0.225]
```

## Troubleshooting Common Issues

### Issue: "Module not found" errors
```yaml
# Solution: Use _requires_ to import modules
_requires_:
    - "$import sys; sys.path.append('.')"

project: ./my_project  # Ensure project path is correct
```

### Issue: Reference not resolving
```yaml
# Wrong: Using # with Python attributes
params: "$@system#model#parameters()"  # ❌

# Correct: Use . for Python attributes
params: "$@system#model.parameters()"  # ✅
```

### Issue: Circular references
```yaml
# Avoid circular references by using lazy evaluation
system:
    model:
        _target_: MyModel
        optimizer_lr: "@system#optimizer#lr"  # ❌ Circular!

    optimizer:
        _target_: torch.optim.Adam
        lr: "$@system#model.get_lr()"  # ❌ Circular!

# Solution: Use vars or computed values
vars:
    lr: 0.001

system:
    model:
        _target_: MyModel
        lr: "%vars#lr"  # ✅

    optimizer:
        _target_: torch.optim.Adam
        lr: "%vars#lr"  # ✅
```

## Best Practices

1. **Organize configs modularly**: Separate base, model, data, and training configs
2. **Use variables**: Define commonly used values in `vars` section
3. **Document your configs**: Add comments explaining non-obvious choices
4. **Version control**: Track config changes alongside code
5. **Validate early**: Test configs with `--trainer#fast_dev_run=true`
6. **Use type hints**: When creating custom classes, use type hints for better IDE support

## Recap and Next Steps

This guide covered the comprehensive configuration system in Lighter. Key takeaways:

*   **Quick Reference**: Symbols (`_target_`, `%`, `@`, `$`, `#`, `.`) provide powerful configuration capabilities
*   **Structure**: Mandatory `trainer` and `system` sections, with optional `_requires_`, `vars`, `args`, and `project`
*   **Flexibility**: Override from CLI, merge configs, use Python expressions
*   **Patterns**: Reusable recipes for common scenarios
*   **Best Practices**: Modular organization and validation

With these configuration skills, you can create sophisticated, maintainable experiment definitions.

## Related Guides
- [Flows Guide](flows.md) - Defining the data flow
- [Project Module](project_module.md) - Custom components
- [Troubleshooting](troubleshooting.md) - Config error solutions
- [Run Guide](run.md) - Running experiments
