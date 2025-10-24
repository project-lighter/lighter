# Project Modules: Your Code, Your Way

Lighter isn't just a framework—it's a platform for your innovation. This guide shows you how to seamlessly integrate custom code while maintaining clean, reusable project structure.

## Why Project Modules? 🎯

Project modules let you seamlessly integrate your custom code into Lighter:

- **🧠 Custom Models** - Your neural network architectures
- **📦 Custom Datasets** - Your data loading logic
- **🎯 Custom Metrics** - Your evaluation methods
- **🔄 Custom Transforms** - Your data preprocessing
- **🎛️ Custom Callbacks** - Your training hooks

**Key Benefits:**

- 📦 **Encapsulation**: Keep research code organized
- ♾️ **Reusability**: Share modules across experiments
- 🚀 **Rapid Prototyping**: Test ideas without framework modifications
- 🌐 **Collaboration**: Easy to share and version control

## Project Structure

Your project folder can be named and located however and wherever you want. You only need to ensure that any folder that is a Python module contains `__init__.py`. In the example below, we see that the project root `my_project` contains `__init__.py` file, just like the `models` and `datasets` subdirectories. On the other hand, the `experiments` directory does not contain any Python modules, so it does not need an `__init__.py` file.

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

!!! warning "Relative Path Behavior"
    The `project` path is relative to your **current working directory** when running the `lighter` command, not relative to the config file location.

    ```bash
    # If you run from parent directory
    cd /path/to/parent && lighter fit /path/to/my_project/experiments/config.yaml --project=/path/to/my_project/

    # If you run from project directory
    cd /path/to/parent/my_project && lighter fit experiments/config.yaml --project=.
    ```

    **Tip:** Use absolute paths to avoid confusion, or be mindful of your current working directory.

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


## Practical Example: Custom Model Architecture

```python
# my_project/models/custom_unet.py
import torch
import torch.nn as nn

class CustomUNet(nn.Module):
    """U-Net for segmentation tasks."""
    def __init__(self, in_channels=3, num_classes=2, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features[:-1]):
            self.decoder.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._block(feature*2, feature))

        self.final = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip_connections = []

        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)

        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skip_connections[idx//2]
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx+1](x)

        return self.final(x)
```

Use in config:
```yaml
project: my_project/

system:
    model:
        _target_: project.models.custom_unet.CustomUNet
        in_channels: 3
        num_classes: 10
        features: [64, 128, 256, 512]
```

## Best Practices for Project Organization 🏆

### Recommended Structure
```
my_project/
├── __init__.py
├── models/           # Neural network architectures
├── datasets/         # Data loading and processing
├── metrics/          # Custom evaluation metrics
├── callbacks/        # Training callbacks
└── utils/            # Helper functions
```

### Key Guidelines

1. **Always include `__init__.py`** in each directory
2. **Use type hints** for better IDE support
3. **Write tests** for critical components
4. **Document with docstrings** for team collaboration
5. **Keep modules focused** - one concept per file

## Running with Custom Modules

```bash
# Basic training
lighter fit config.yaml

# With module path override
lighter fit config.yaml --project=./my_research_project

# Multiple configs with custom modules
lighter fit base.yaml,models/unet.yaml,data/custom.yaml
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **ModuleNotFoundError** | Check `__init__.py` files and project path |
| **AttributeError** | Import classes in `__init__.py` |
| **Circular imports** | Use lazy imports inside functions |
| **Path issues** | Use absolute imports or `project: ./my_project` |

## Recap and Next Steps

You're now equipped to build sophisticated custom modules:

🎯 **Key Takeaways:**

- Structure projects with clear module organization
- Use type hints and documentation for maintainability
- Leverage advanced patterns (multi-modal, caching, custom augmentations)
- Test your modules for reliability

💡 **Remember:** Great research code is modular, tested, and reusable!

## Related Guides
- [Configuration](configure.md) - Referencing project modules
- [Adapters](adapters.md) - Custom adapter creation
- [Metrics](metrics.md) - Custom metric creation
