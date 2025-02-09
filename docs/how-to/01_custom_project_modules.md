# How to Use Custom Project Modules in Lighter

## Introduction to Custom Modules

Lighter's extensibility allows seamless integration of your custom modules (e.g., models, datasets, callbacks, metrics). This guide shows how to use them to tailor Lighter to your needs.

Custom modules are Python modules you create within your project, unlike Lighter or external libraries' modules. Benefits include:

*   **Encapsulation**: Organize project-specific logic (models, datasets) in dedicated modules for maintainability.
*   **Flexibility**: Extend Lighter's functionality to fit your research.
*   **Prototyping**: Quickly experiment with new ideas by integrating custom modules.

## Project Structure 

Your project folder can be named and located however and wherever you want. You only need to ensure that any folder that is a Python modules contains `__init__.py`. In the example below, we see that the project root `my_project` contains `__init__.py` file, just like the `models` and `datasets` subdirectories. On the other hand, the `experiments` directory does not contain any Python modules, so it does not need an `__init__.py` file.

```
my_project/
├── __init__.py
├── models/
│   ├── __init__.py
│   └── my_model.py
├── datasets/
│   ├── __init__.py
│   └── my_dataset.py
└── experiments/
    ├── finetune_full.yaml
    └── finetune_decoder.yaml
```


## Defining Project Modules

Within the module directories (e.g., `models/`, `datasets/`), you define your custom Python modules as regular Python files (`.py`). Let's define a custom model `MyModel` in `my_model.py`

```python title="my_project/models/my_model.py"
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # Define a linear layer
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        # Forward pass through the linear layer
        return self.linear(x)
```

and a custom dataset `MyDataset` in `my_dataset.py`:

```python title="my_project/datasets/my_dataset.py"
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        # Load data from data_path (implementation not shown)
        self.samples = [...] # List of data samples (replace [...] with actual data loading)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Preprocess sample (implementation not shown)
        # ... 
        if self.transform:
            sample = self.transform(sample)
        return sample
```

## Importing Project Modules in the Config

To use your project's modules in Lighter, you need to define it in your config. Lighter's dynamic module loading mechanism, powered by the [`import_module_from_path`](../../reference/utils/dynamic_imports/#lighter.utils.dynamic_imports.import_module_from_path) function, will then import that folder as a module named `project`.

### Specifying `project` Path

Specify your project's root directory in `config.yaml` using the `project` key. This tells Lighter where to find your custom modules.

**Example:**

```yaml title="config.yaml"
project: my_project/ # Project root path
```

### Referencing Project Modules

Reference your projects modules just like you reference any other module. For example, look at `system#model` and `system#dataloaders#train#dataset`:

**Example:**

```yaml title="config.yaml" hl_lines="5 13"
project: my_project/

system:
  model:
    _target_: project.models.MyModel
    input_size: 784
    num_classes: 10

  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: project.datasets.MyDataset
        data_path: "data/train.csv"
        # ... dataset arguments ...
      batch_size: 32
      shuffle: True
```


## Running Lighter with Custom Modules

To run your Lighter experiments that use custom modules, you simply execute the `lighter fit` command (or other Lighter CLI commands) with your `config.yaml` file, just as you would with built-in modules.

**Example: Running Training with Custom Modules**

```bash title="Terminal"
lighter fit --config config.yaml
```

As long as your `config.yaml` file correctly specifies the `project` path and the `_target_` paths to your custom modules, Lighter will dynamically load and use them during the experiment execution.

## Recap: Steps to Use Custom Modules

1.  Organize project with clear directory structure (e.g., subdirectories for modules).
2.  Define custom modules (models, datasets) as Python files in project directories.
3.  Specify `project` path in `config.yaml`.
4.  Reference modules in `config.yaml` using `_target_` with project-relative paths.
5.  Run Lighter as usual.

These steps enable seamless integration of custom code, leveraging Lighter's flexibility for customized deep learning systems.

Next, explore the [How-To guide on Debugging Configuration Errors](02_debugging_config_errors.md) to learn how to troubleshoot common configuration issues, or return to the [How-To guides section](../how-to/01_custom_project_modules.md) for more practical problem-solving guides. You can also go back to the [Design section](../design/01_overview.md) for more conceptual documentation or the [Tutorials section](../tutorials/01_configuration_basics.md) for end-to-end examples.
