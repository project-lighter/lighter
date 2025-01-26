# Quickstart 
Get started with Lighter in just a few minutes! This guide will walk you through the installation and setup process, enabling you to quickly configure and run your experiments.

## Installation

Latest release (recommended):
```python
pip install lighter
```

Up-to-date with the main branch (may contain experimental features):
```python
pip install lighter --pre
```


## Building a config
At the heart of the Lighter ecosystem is a YAML configuration file. This file acts as the central hub for managing your experiments, allowing you to define, adjust, and control every aspect without delving into the underlying code.

A Lighter config contains two main components:

- **Trainer**: Manages the training loop and related settings.
- **System**: Defines the model, datasets, optimizer, and other components.

### Trainer
The Trainer section encapsulates all the details necessary for running training, evaluation, or inference processes. It is a vital component of training automation in PyTorch Lightning. For more details, refer to the [PyTorch Lightning Trainer documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html).

In this section, you can configure various parameters such as the number of epochs, GPUs, nodes, and more. Any parameter available in the `pytorch_lightning.Trainer` class can be specified here. For a comprehensive list, see the [API documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api).

Defining this in our config looks something like this
```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 100
```

For more information see [here](./config.md)

### System
The System component encompasses all elements of a deep learning setup, including the model, optimizer, criterion, datasets, and metrics. This is where the core logic of building deep learning models is defined. System is highly adaptable, supporting tasks like classification, segmentation, object detection, and self-supervised learning. It provides a structured approach to defining each component, akin to writing your code with a clear framework.

This structure offers powerful extensibility, allowing training experiments for classification and self-supervised learning to follow a consistent template. Below is an example of a System configuration for training a supervised classification model on the CIFAR10 dataset:

```yaml
system:
  _target_: lighter.System
  batch_size: 512

  model:
    _target_: torchvision.models.resnet18
    num_classes: 10

  criterion:
    _target_: torch.nn.CrossEntropyLoss

  optimizer:
    _target_: torch.optim.Adam
    params: "$@system#model.parameters()"
    lr: 0.001

  datasets:
    train:
      _target_: torchvision.datasets.CIFAR10
      download: True
      root: .datasets
      train: True
      transform:
        _target_: torchvision.transforms.Compose
        transforms:
          - _target_: torchvision.transforms.ToTensor
          - _target_: torchvision.transforms.Normalize
            mean: [0.5, 0.5, 0.5]
            std: [0.5, 0.5, 0.5]
      # Format the batch as required by Lighter.
          - _target_: torchvision.transforms.Lambda
            lambd: $lambda x: {"input": x[0], "target": x[1]}'

```

5.  Postprocessing functions can be defined for various stages, such as batch processing, criterion evaluation, metrics calculation, and logging. These functions enable data modification at different points in the workflow, enhancing flexibility and control.

For more information about each of the System components and how to override them, see [here](./config.md).

## Running this experiment with Lighter
To run an experiment with Lighter, combine the Trainer and System configurations into a single YAML file and execute the following command in your terminal:

=== "cifar10.yaml"
    ```yaml
    trainer:
      _target_: pytorch_lightning.Trainer
      max_epochs: 100
      
    system:
      _target_: lighter.System
      batch_size: 512

      model:
        _target_: torchvision.models.resnet18
        num_classes: 10

      criterion:
        _target_: torch.nn.CrossEntropyLoss

      optimizer:
        _target_: torch.optim.Adam
        params: "$@system#model.parameters()"
        lr: 0.001

      datasets:
        train:
          _target_: torchvision.datasets.CIFAR10
          download: True
          root: .datasets
          train: True
          transform:
            _target_: torchvision.transforms.Compose
            transforms:
              - _target_: torchvision.transforms.ToTensor
              - _target_: torchvision.transforms.Normalize
                mean: [0.5, 0.5, 0.5]
                std: [0.5, 0.5, 0.5]

      postprocessing:
          # Ensure the batch is formatted as a dictionary with 'input' and 'target' keys.
          batch:
              train: '$lambda x: x'

    ```
=== "Terminal"
    ```
    lighter fit --config cifar10.yaml
    ```


Congratulations! You've successfully run your first training example using Lighter.
