# Deep Dive into Lighter's Configuration System

## Introduction to Configuration

Configuration is key to Lighter, enabling declarative management of deep learning experiments via human-readable YAML files. Benefits include:

*   **Reproducibility**: Explicitly capture all settings for reliable experiment reproduction.
*   **Simplified Management**: Manage complex experiments with a single `config.yaml`.
*   **Parameter Sweeping**: Easily modify and iterate by changing config values.
*   **Collaboration**: Share and discuss experiments via config files.

This document details Lighter's configuration system, covering its structure, features, and effective usage.

## YAML Structure of `config.yaml`

Lighter configurations use YAML, a human-readable format. `config.yaml` is hierarchical, using key-value pairs in sections.

**Top-Level Keys**:

Typical `config.yaml` top-level keys:

*   **`trainer`**: PyTorch Lightning `Trainer` configuration (training process settings).
*   **`system`**: `lighter.System` configuration (deep learning system definition):
    *   `model`: Neural network model.
    *   `dataloaders`: Data loaders (train, val, test, predict).
    *   `optimizer`: Optimizer algorithm.
    *   `scheduler`: Learning rate scheduler.
    *   `criterion`: Loss function.
    *   `metrics`: Evaluation metrics.
    *   `adapters`: Data handling and argument adaptation.
    *   `inferer`: Inference control.
*   **`project`**: (Optional) Project directory path (for dynamic module loading of custom modules).
*   **`args`**: (Optional) Command-line config overrides.

**Example `config.yaml` Structure**:

```yaml title="config.yaml"
trainer: # PyTorch Lightning Trainer configuration
  _target_: pytorch_lightning.Trainer
  max_epochs: 100
  accelerator: "auto"
  devices: 1

system: # lighter.System configuration
  _target_: lighter.System

  model: # Model definition
    _target_: my_project.models.MyModel # Path to custom model class
    input_size: [1, 28, 28]
    num_classes: 10

  dataloaders: # DataLoaders configuration
    train:
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: my_project.datasets.MyDataset # Path to custom dataset class
        data_path: "data/train.csv"
      batch_size: 32
      shuffle: true
    val: # Validation dataloader
      _target_: torch.utils.data.DataLoader
      # ... (validation dataloader config) ...

  optimizer: # Optimizer configuration
    _target_: torch.optim.Adam
    lr: 1.0e-3
    weight_decay: 1.0e-5

  criterion: # Loss function configuration
    _target_: torch.nn.CrossEntropyLoss

  metrics: # Metrics configuration
    train:
      - _target_: torchmetrics.Accuracy
        task: "multiclass"
        num_classes: 10
    val:
      - _target_: torchmetrics.Accuracy
        # ... (accuracy metric config) ...

project: my_project/ # Project path for dynamic module loading

args: # Command-line argument overrides
  fit: # 'lighter fit' stage arguments
    trainer:
      max_epochs: 20 # Override max_epochs to 20 when using 'lighter fit'
```

## MONAI Bundle Style Configuration

Lighter adopts the configuration style from MONAI Bundle, which is designed for modularity and flexibility in defining deep learning components. This style relies heavily on the `_target_` key to specify Python classes and their arguments.

**`_target_` Key**:

The `_target_` key is used throughout Lighter configurations to indicate the Python class that should be instantiated. The value of `_target_` is a string representing the Python path to the class (e.g., `torch.optim.Adam`, `my_project.models.MyModel`).

**Example: Defining a Model**:

```yaml
system:
  model:
    _target_: my_project.models.MyModel # Path to custom model class
    input_size: [1, 28, 28]           # Argument for MyModel.__init__
    num_classes: 10              # Argument for MyModel.__init__
```

In this example:

*   `_target_: my_project.models.MyModel` tells Lighter to instantiate the `MyModel` class from the `my_project.models` module.
*   `input_size: [1, 28, 28]` and `num_classes: 10` are passed as keyword arguments to the `MyModel` class's `__init__` constructor.


**Benefits of `_target_`**:

*   **Declarative**: Config defines classes and arguments in YAML, not code.
*   **Dynamic**: Flexible, modular setups via dynamic class instantiation.
*   **Customizable**: Integrates custom modules by specifying Python paths, enhancing modularity.
*   **Readable**: Clear mapping to Python class instantiation improves config readability.
## Stage-Specific Configurations

Lighter uses the concept of **stages** to manage configurations for different phases of a deep learning workflow. The stages are:

*   **`fit`**: Training and validation stage (`lighter fit`).
*   **`validate`**: Validation stage (`lighter validate`).
*   **`test`**: Testing stage (`lighter test`).
*   **`predict`**: Prediction stage (`lighter predict`).

**Configuration Pruning**:

When you run Lighter for a specific stage (e.g., `lighter fit`), Lighter **prunes** the configuration file, keeping only the relevant sections for that stage. This ensures that only necessary components are initialized and used, optimizing resource usage and avoiding conflicts.

**Example: Stage-Specific DataLoaders**:

```yaml
system:
  dataloaders:
    train: # Dataloader for 'fit' stage
      _target_: torch.utils.data.DataLoader
      # ... (train dataloader config) ...
    val: # Dataloader for 'fit' and 'validate' stages
      _target_: torch.utils.data.DataLoader
      # ... (validation dataloader config) ...
    test: # Dataloader for 'test' stage
      _target_: torch.utils.data.DataLoader
      # ... (test dataloader config) ...
    predict: # Dataloader for 'predict' stage
      _target_: torch.utils.data.DataLoader
      # ... (predict dataloader config) ...
```

In this example, different dataloaders are defined for `train`, `val`, `test`, and `predict` stages. When you run `lighter fit`, Lighter will use the `train` and `val` dataloaders. When you run `lighter validate`, it will use the `val` dataloader. Similarly, `lighter test` will use `test`, and `lighter predict` will use `predict`.

**`Resolver` Class**:

Lighter uses a `Resolver` class (in `lighter/engine/resolver.py`) to handle configuration loading, parsing, and stage-specific pruning. The `Resolver` is responsible for:

1.  **Loading the `config.yaml` file**.
2.  **Validating the configuration** against a predefined schema (using Cerberus, see below).
3.  **Resolving `_target_` paths** and dynamically importing modules.
4.  **Pruning the configuration** based on the current stage.
5.  **Instantiating configured objects** (models, datasets, etc.).

## Validation with Cerberus

Lighter validates `config.yaml` against a schema using Cerberus. Schema defines config structure, types, and fields.

**`SCHEMA` in `engine/config.py`**:

Config schema is in `SCHEMA` dict in `lighter/engine/config.py`. It specifies valid config structure and rules.

**Example Schema Snippet (from `engine/config.py`)**:

```python
SCHEMA = {
    "trainer": { # Schema for 'trainer' section
        "type": "dict",
        "required": False,
        "schema": {
            "_target_": {"type": "string", "required": True}, # '_target_' is required
            "max_epochs": {"type": "integer", "required": False, "min": 1}, # 'max_epochs' is optional, must be integer >= 1
            "accelerator": {"type": "string", "required": False}, # 'accelerator' is optional, must be string
            # ... (other trainer schema rules) ...
        },
    },
    "system": { # Schema for 'system' section
        "type": "dict",
        "required": True, # 'system' section is required
        "schema": {
            "_target_": {"type": "string", "required": True}, # '_target_' is required
            "model": {"type": "dict", "required": True}, # 'model' is required, must be dict
            "dataloaders": {"type": "dict", "required": True}, # 'dataloaders' is required, must be dict
            "optimizer": {"type": "dict", "required": True}, # 'optimizer' is required, must be dict
            "scheduler": {"type": "dict", "required": False}, # 'scheduler' is optional
            "criterion": {"type": "dict", "required": True}, # 'criterion' is required
            "metrics": {"type": "dict", "required": False}, # 'metrics' is optional
            "adapters": {"type": "dict", "required": False}, # 'adapters' is optional
            "inferer": {"type": "dict", "required": False}, # 'inferer' is optional
            # ... (other system schema rules) ...
        },
    },
    # ... (schema for other top-level keys) ...
}
```

**Validation Process**:

Lighter validates `config.yaml` against `SCHEMA`. Invalid configs raise `ConfigurationError` with detailed messages for quick issue resolution.

**Benefits of Validation**:

*   **Early Error Detection**: Catch errors early, before runtime.
*   **Configuration Consistency**: Ensure consistent config structure and rules.
*   **Improved User Experience**: Helpful error messages for valid config creation.

## Overriding Configuration from CLI

Override config values from CLI using `args` section in `config.yaml` for commands like `lighter fit`.

**`args` Section**:

Optional top-level `args` in `config.yaml` defines stage-specific overrides.

**Example `args` Section**:

```yaml title="config.yaml"
args:
  fit: # Arguments for 'lighter fit' stage
    trainer: # Override 'trainer' section
      max_epochs: 20 # Override 'trainer.max_epochs' to 20
      accelerator: "gpu" # Override 'trainer.accelerator' to "gpu"
    system: # Override 'system' section
      optimizer:
        lr: 1.0e-4 # Override 'system.optimizer.lr' to 1.0e-4
  validate: # Arguments for 'lighter validate' stage
    trainer:
      devices: 2 # Override 'trainer.devices' to 2 for validation
```

**CLI Overrides**:

Lighter CLI commands with `config.yaml` apply `args` overrides.

**Example Usage**:

```bash title="Terminal"
lighter fit --config config.yaml # 'fit' stage with 'args.fit' overrides
lighter validate --config config.yaml # 'validate' stage with 'args.validate' overrides
```

In the `config.yaml` example:

*   `lighter fit`: Overrides `trainer.max_epochs` to `20`, `trainer.accelerator` to `"gpu"`, `system.optimizer.lr` to `1.0e-4`.
*   `lighter validate`: Overrides `trainer.devices` to `2`.

**Order of Precedence**:

CLI overrides have **highest precedence** over `args` section.

**Benefits of CLI Overrides**:

*   **Flexibility**: Modify settings without config file edits.
*   **Parameter Tuning**: Run sweeps/ablations by overriding CLI parameters.
*   **Stage-Specific Adjustments**: Stage-specific settings (e.g., more devices for validation).

## Advanced Configuration Features

Lighter's configuration system offers several advanced features for more complex and flexible experiment setups:

*   **Nested Configurations**: Organize complex configurations hierarchically using nested dictionaries and lists in `config.yaml`.
*   **Environment Variables**: (Planned) Support for environment variables in configurations for sensitive/dynamic settings.
*   **Custom Resolvers**: (Planned) Define custom resolvers for advanced configuration logic.

## Recap: Mastering Configuration

Lighter's configuration system streamlines deep learning experiments. YAML structure, `_target_` style, stage-specific configs, Cerberus validation, and CLI overrides enable efficient experiment management, reproducibility, and focused research.
