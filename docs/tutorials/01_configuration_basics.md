# Configuration Basics

## Config Structure

### Mandatory Sections

Lighter uses YAML configs to define experiments. A typical config is organized into two mandatory sections:

*   **`trainer`**: Configures [`Trainer` (PyTorch Lightning)](https://lightning.ai/docs/pytorch/stable/common/trainer.html).
*   **`system`**: Defines [`System` (Lighter)](../../reference/system/#lighter.system.System) that encapsulates components such as model, criterion, optimizer, or dataloaders.

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
*   **`vars`**: Store variables for use in other parts of the config. Useful to avoid repetition and easily update values. See [Referencing Other Components](#referencing-other-components).
*   **`args`**: Arguments to pass to the the stage of the experiment being run. See [Stages](#stages).
*   **`project`**: Path to your project directory. Used to import custom modules. For more details, see [Custom Project Modules](../how-to/01_custom_project_modules.md).


## Stages

Lighter operates in *stages*: `fit`, `validate`, `test`, `predict`, `lr_find`, and `scale_batch_size`.

If you're familiar with PyTorch Lightning, you will notice that the stages correspond to methods from `Trainer` (`fit`, `validate`, `test`, `predict`) and `Tuner` (`lr_find`, `scale_batch_size`). For reference, see the PyTorch Lightning [Trainer](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer) and [Tuner](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.tuner.tuning.Tuner.html#lightning.pytorch.tuner.tuning.Tuner) documentation.

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

!!! note "Difference between `#` and `.`"

    Use `#` to reference elements of the config, and `.` to access attributes/methods of Python objects. For example, `"$@system#model.parameters()"` fetches the model instance from `@system#model` and runs `.parameters()` on it as indicated by `$`.


### Overriding Config from CLI

Any parameter in the config can be overridden from the command line. For example, to change `max_epochs` in `trainer` from `10` to `20`:

```bash
lighter fit config.yaml --trainer#max_epochs=20
```

To override an element of a list, simply refer its index:

```bash
lighter fit config.yaml --trainer#callbacks#1#monitor="val_loss"
```
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

### Merging Configs

You can merge multiple configs by combining them with `,`. For example, to merge two configs `config1.yaml` and `config2.yaml`:

```bash
lighter fit config1.yaml,config2.yaml
```

This will merge the two configs, with the second config overriding any conflicting parameters from the first one.

## Recap and Next Steps

This section covered the fundamental aspects of configuring experiments with Lighter using YAML files.  Key takeaways include:

*   **Structure:** Lighter configs are structured into mandatory `trainer` and `system` sections, with optional sections like `_requires_`, `vars`, `args`, and `project`.
*   **Stages:** Lighter operates in stages (e.g., `fit`, `validate`, `test`), each configurable via the `args` section.
*   **Config Syntax:** Lighter leverages the MONAI Bundle configuration system, using `_target_` to instantiate classes and key-value pairs for arguments.
*   **Referencing:** Components can be referenced using `%` (textual replacement) or `@` (evaluated Python value).  Understanding the difference is crucial for correct instantiation and interaction of components.
*   **Python Expressions:** Python expressions can be evaluated within the config using `$`. This is frequently used in conjunction with referencing (e.g., `"$@system#model.parameters()"`).
*   **CLI Overrides:** Any config parameter can be overridden from the command line, providing flexibility for experimentation.
*   **Config Merging:** Multiple configs can be merged using commas, allowing for modularity and reuse.

By mastering these configuration basics, you can effectively define and manage your Lighter experiments.

Next tutorials: [image classification](02_image_classification.md), [semantic segmentation](03_semantic_segmentation.md). See [How-To guides](../how-to/02_debugging_config_errors.md) for debugging, [Design section](../design/02_configuration_system.md) for deeper dive.
