# Quickstart 
Get up and running in under 5 mins! 

## Installation

```python
pip install project-lighter
```

For bleeding edge version, run
```python
pip install project-lighter --pre
```


## Building a config
Key to the Lighter ecosystem is a YAML file that serves as the central point of control for your experiments. It allows you to define, manage, and modify all aspects of your experiment without diving deep into the code. 

A Lighter config contains two main components: 

- Trainer
- LighterSystem

### Trainer
Trainer contains all the information about running a training/evaluation/inference process and is a crucial component of training automation in Pytorch Lightning. Please refer to the [Pytorch Lightning's Trainer documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html) for more information.

Simply put, you can set several things here such as the number of epochs, the number of gpus, the number of nodes, etc. All the parameters that can be set in the `pytorch_lightning.Trainer` class can also go here. Please refer to the [API](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api) for more information.

Defining this in our config looks something like this
```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 100
```

For more information see [here](./config.md)

### LighterSystem
LighterSystem encapsulates all parts of a deep learning setup, such as the model, optimizer, criterion, datasets, metrics, etc. This is where the "science" of building deep learning models is developed. The LighterSystem is highly flexible and contain logic suitable for any task - classification, segmentation, object detection, self-supervised learning, etc. Think of this as writing your code but with predefined structure on where to define each compoenent (such as model, criterion, etc.)

This provides powerful extensibility as training experiments for classification and self-supervised learning can follow a similar template. An example of a LighterSystem for training a supervised classification model on CIFAR10 dataset is shown below,

```yaml
system:
  _target_: lighter.LighterSystem
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
```

For more information about each of the LighterSystem components and how to override them, see [here](./config.md)

## Running this experiment with Lighter
We just combine the Trainer and LighterSystem into a single YAML and run the command in the terminal as shown,

=== "cifar10.yaml"
    ```yaml

    trainer:
      _target_: pytorch_lightning.Trainer
      max_epochs: 100

    system:
      _target_: lighter.LighterSystem
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

    ```
=== "Terminal"
    ```
    lighter fit --config_file cifar10.yaml
    ```


Congratulations!! You have run your first training example with Lighter.