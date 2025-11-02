# Image Classification with CIFAR-10

This tutorial shows you how to train a CNN on CIFAR-10 using Lighter.

**What you'll build:**

- A custom CNN model
- Complete training configuration
- Working image classification pipeline

## Step 1: Project Structure

Create this folder structure:

```bash
image_classification/
├── __init__.py             # Makes it a Python module
├── experiments/
│   └── config.yaml         # Configuration file
└── models/
    ├── __init__.py
    └── simple_cnn.py       # Your model
```

**Quick setup (Unix/Linux/Mac):**
```bash
mkdir -p image_classification/{models,experiments}
touch image_classification/__init__.py \
      image_classification/models/{__init__.py,simple_cnn.py} \
      image_classification/experiments/config.yaml
```

!!! tip
    The `__init__.py` files are essential - they make folders importable as Python modules.


## Step 2: Create the Model

Save this CNN model in `image_classification/models/simple_cnn.py`:

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 8 * 8, num_classes) # Assuming 32x32 images

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x
```

## Step 3: Complete Configuration

Save this in `image_classification/experiments/config.yaml`:

```yaml
project: .  # Use '.' if running from image_classification folder

trainer:
    _target_: pytorch_lightning.Trainer
    accelerator: "auto"  # Use GPU if available, else CPU
    max_epochs: 10

system:
    _target_: lighter.System

    model:
        _target_: project.models.simple_cnn.SimpleCNN
        num_classes: 10

    criterion:
        _target_: torch.nn.CrossEntropyLoss

    optimizer:
        _target_: torch.optim.Adam
        params: "$@system#model.parameters()"  # Link to model's learnable parameters
        lr: 1.0e-3

    metrics:
        train:
            - _target_: torchmetrics.Accuracy
              task: "multiclass"
              num_classes: 10
        test: "%#train"

    flows:
        train:
            _target_: lighter.flow.Flow
            batch: ["input", "target"]
            model: ["input"]
            criterion: ["pred", "target"]
            metrics: ["pred", "target"]
        test:
            _target_: lighter.flow.Flow
            batch: ["input", "target"]
            model: ["input"]
            metrics: ["pred", "target"]

    dataloaders:
        train:
            _target_: torch.utils.data.DataLoader
            batch_size: 32
            shuffle: True
            num_workers: 4
            dataset:
                _target_: torchvision.datasets.CIFAR10
                root: cifar10/
                download: True
                train: True
                transform:
                    _target_: torchvision.transforms.Compose
                    transforms:
                        - _target_: torchvision.transforms.ToTensor
                        - _target_: torchvision.transforms.Normalize
                          mean: [0.5, 0.5, 0.5]
                          std: [0.5, 0.5, 0.5]
        test:
            _target_: torch.utils.data.DataLoader
            batch_size: 32
            num_workers: 4
            dataset:
                _target_: torchvision.datasets.CIFAR10
                root: cifar10/
                download: True
                train: False
                transform:
                    _target_: torchvision.transforms.Compose
                    transforms:
                        - _target_: torchvision.transforms.ToTensor
                        - _target_: torchvision.transforms.Normalize
                          mean: [0.5, 0.5, 0.5]
                          std: [0.5, 0.5, 0.5]
```

!!! info "Path Configuration"
    - Use `project: .` if running from inside `image_classification/`
    - Use `project: ./image_classification` if running from parent directory
    - Or use absolute path: `project: /home/user/image_classification`

## Step 4: Train the Model

Navigate to your project folder and run:

```bash
cd image_classification
lighter fit experiments/config.yaml
```

You'll see training progress with loss and accuracy metrics.

## Step 5: Test the Model

Evaluate on the test set:

```bash
lighter test experiments/config.yaml
```

## Tips & Tricks

**Quick experiments:**
```bash
# Change epochs from CLI
lighter fit experiments/config.yaml --trainer#max_epochs=20

# Use GPU if available
lighter fit experiments/config.yaml --trainer#accelerator=gpu

# Fast debugging (2 batches only)
lighter fit experiments/config.yaml --trainer#fast_dev_run=2
```

**Common issues:**

- **ModuleNotFoundError**: Check that all folders have `__init__.py`
- **Wrong project path**: Use `pwd` to check current directory
- **CIFAR-10 download fails**: Check internet connection

## What's Next?

✓ You've trained a CNN on CIFAR-10!

✓ You understand project structure and configuration

✓ You can run training and testing

Explore more:

- [Configuration Guide](../how-to/configure.md) - Advanced config features
- [Custom Metrics](../how-to/metrics.md) - Add custom evaluation metrics
- [Flows](../how-to/flows.md) - Handle complex data flows
