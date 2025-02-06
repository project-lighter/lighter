# How to Use Custom Project Modules in Lighter

## Introduction to Custom Modules

Lighter's extensibility allows seamless integration of your custom modules (e.g., models, datasets, callbacks, metrics). This guide shows how to use them to tailor Lighter to your needs.

Custom modules are Python modules you create within your project, unlike built-in Lighter modules or external libraries. Benefits include:

*   **Encapsulation**: Organize project-specific logic (models, datasets) in dedicated modules for maintainability.
*   **Reusability**: Reuse custom modules across experiments, reducing code duplication.
*   **Version Control**: Track changes and collaborate easily with project-local modules.
*   **Flexibility**: Extend Lighter's functionality to fit your research.

## Project Structure for Custom Modules

For effective use of custom modules, organize your project clearly. A typical Lighter project structure is:

```
my_project/
├── config.yaml          
├── models/             
│   └── my_model.py     
├── datasets/           
│   └── my_dataset.py   
├── callbacks/          
├── metrics/            
└── utils/              
```

*   **`my_project/`**: Root directory.
*   **`config.yaml`**: Lighter configuration file.
*   **`models/`, `datasets/`, `callbacks/`, `metrics/`, `utils/`**: Subdirectories for custom modules (optional).

A text-based tree view of this structure:

```
my_project/
├── config.yaml
├── models/
│   └── my_model.py
├── datasets/
│   └── my_dataset.py
├── callbacks/
├── metrics/
└── utils/
```

In this structure:

*   **`my_project/`**: This is the root directory of your Lighter project. You can name it according to your project's purpose.
*   **`models/`**, **`datasets/`**, **`callbacks/`**, **`metrics/`**, **`utils/`**: These are subdirectories within your project where you can organize your custom Python modules based on their type (models, datasets, callbacks, metrics, utility functions, etc.). You can create or omit these directories as needed, depending on your project's complexity and the types of custom modules you are using.
*   **`my_model.py`**, **`my_dataset.py`**, etc.: These are example Python files where you define your custom classes, functions, and logic for models, datasets, etc. You can create multiple Python files within each subdirectory to further organize your custom modules.
*   **`config.yaml`**: This is your Lighter configuration file, where you will specify how to use your custom modules.
*   **`train.py`** (optional): This is an optional entry point script that you can create to run your Lighter training workflows. You can also run Lighter directly from the command line using `lighter fit config.yaml`.

## Defining Custom Modules

Within the module directories (e.g., `models/`, `datasets/`), you define your custom Python modules as regular Python files (`.py`). Each file can contain one or more classes, functions, and variables, depending on your needs.

**Example: Custom Model (`my_project/models/my_model.py`)**

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

**Example: Custom Dataset (`my_project/datasets/my_dataset.py`)**

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

These are just simple examples. Your custom modules can be as complex as needed for your project.

## Importing Custom Modules in `config.yaml`

To use your custom modules in Lighter, you need to reference them in your `config.yaml` file using the `_target_` key. Lighter's dynamic module loading mechanism, powered by the `import_module_from_path` function, will then import and instantiate your custom modules at runtime.

### Specifying `project` Path

Specify your project's root directory in `config.yaml` using the `project` key. This tells Lighter where to find your custom modules.

**Example:**

```yaml title="config.yaml"
project: my_project/ # Project root path
```

### Referencing Modules with `_target_`

Reference custom modules using `_target_` with the Python path relative to your project root.

**Example:**

```yaml title="config.yaml"
project: my_project/

system:
  model:
    _target_: project.models.MyModel # Load custom model
    input_size: 784
    num_classes: 10

  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: project.datasets.MyDataset # Load custom dataset
        data_path: "data/train.csv"
        # ... dataset arguments ...
      batch_size: 32
      shuffle: True
```

In this setup, `_target_: project.models.MyModel` loads `MyModel` from `my_project/models/my_model.py`. Lighter prepends the `project` path and uses `import_module_from_path` for dynamic import.

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
