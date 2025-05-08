In this tutorial we will learn how to:

1. Set up the project folder
2. Implement a custom CNN model
3. Define the config for training and testing on CIFAR10 dataset
4. Train and test the model using Lighter

## Setting up the Project

First, create a new project directory named `image_classification` with the following structure:

```plaintext
config.yaml
image_classification/
├── __init__.py
└── models/
    ├── __init__.py
    └── simple_cnn.py
```

!!! warning

    Do not forget the `__init__.py` files. For more details, refer to the [Project Module](../how-to/project_module.md) guide.

## Setting up Dataloaders

`system`'s `dataloaders` section defines dataloaders for `train`, `val`, `test`, and `predict` stages. Let's start by configuring the training dataloader for CIFAR10.

!!! note
    The complete configuration is provided [few sections later](#complete-configuration).

```yaml
system:
# ...
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
```

This is equivalent to the following Python code:

```python
import torch
import torchvision

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = torchvision.datasets.CIFAR10(
    root="cifar10/",
    download=True,
    train=True,
    transform=transforms
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

## Setting up the Model

### Defining a Custom Model

We will use a simple CNN for image classification. Define this model in `image_classification/models/simple_cnn.py`.

```python title="image_classification/models/simple_cnn.py"
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

### Reference the Custom Model in `config.yaml`

Now that we have defined the model, let's specify it in the `config.yaml` file.

```yaml title="config.yaml" hl_lines="1 5"
project: /path/to/image_classification

system:
  model:
    _target_: project.models.simple_cnn.SimpleCNN
    num_classes: 10  # Matches CIFAR10 classes
```

The `project` section tells Lighter where to import the project module from. This allows us to use our `SimpleCNN` class by referencing `project.models.simple_cnn.SimpleCNN`.


## Complete Configuration

Now, let's put together the complete `config.yaml` file for training the `SimpleCNN` on CIFAR10:

```yaml title="config.yaml"
project: /path/to/image_classification

trainer:
    _target_: pytorch_lightning.Trainer
    accelerator: "auto" # Use GPU if available, else CPU
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
        params: "$@system#model.parameters()" # Link to model's learnable parameters
        lr: 1.0e-3

    metrics:
        train:
            - _target_: torchmetrics.Accuracy
              task: "multiclass"
              num_classes: 10
        test: "%#train"

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

This configuration defines all the necessary components for training and testing:

*   **`trainer`**: Configures the PyTorch Lightning Trainer to use automatic accelerator selection and train for a maximum of 10 epochs.
*   **`system`**: Defines the Lighter System.
    *   **`model`**: Specifies the `SimpleCNN` model, a custom model you defined in `image_classification/models/simple_cnn.py`.
    *   **`criterion`**: Sets the loss function to `CrossEntropyLoss`.
    *   **`optimizer`**: Uses the `Adam` optimizer with a learning rate of 1.0e-3.
    *   **`metrics`**: Defines accuracy metrics for training and testing stages.
    *   **`dataloaders`**: Configures `DataLoader`s for `train` and `test` stages, using the CIFAR10 dataset and appropriate transforms.

## Training Execution

To start training, save the above configuration as `config.yaml` in your project directory. Ensure that you have created the `image_classification/models/simple_cnn.py` file as well. Then, open your terminal, navigate to your project directory, and run the following command:

```bash title="Terminal"
lighter fit config.yaml
```

Lighter will parse your `config.yaml`, initialize all the components, and start the training process using PyTorch Lightning. You will see the training progress, including loss and metrics, logged in your terminal.

## Evaluation

After training, you can evaluate your model on the test set:

```bash title="Terminal"
lighter test config.yaml
```

Lighter will load the best checkpoint saved during training (if a `ModelCheckpoint` callback is used in the configuration, which is often the default in more complex setups) and evaluate the model on the specified dataloader, reporting the metrics defined in the `system.metrics` section for the`test` stage, respectively.

## Next Steps

In this tutorial, you have successfully trained and evaluated an image classification model on the CIFAR10 dataset using Lighter.

You now have a solid foundation for building more complex experiments with Lighter. Head over to the [How-To guides](../how-to/project_module.md) to explore Lighter's features in more detail.
