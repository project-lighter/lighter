# Configuration System

Lighter is a configuration-centric framework that uses YAML files to set up machine learning workflows. These configurations cover everything from model architecture selection, loss functions, and optimizers to dataset preparation and the execution of training, evaluation, and inference processes.

Our configuration system is inspired by the MONAI bundle parser, offering a standardized structure. Each configuration requires several mandatory components to be defined.

The configuration is divided into two main components:
- **Trainer**: Handles the training process, including epochs, devices, etc.
- **LighterSystem**: Encapsulates the model, optimizer, datasets, and other components.

Let's explore a simple example configuration to understand Lighter's configuration system better. You can expand each section for more details on specific concepts.

<div class="annotate" markdown>

```yaml title="cifar10.yaml"

trainer:
  _target_ (1): pytorch_lightning.Trainer 
  max_epochs (2): 100

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
    params: "$@system#model.parameters()" (3)
    lr: 0.001

  datasets:
    train:
      _target_: torchvision.datasets.CIFAR10
      download: True
      root: .datasets
      train: True
      transform:
        _target_: torchvision.transforms.Compose
        transforms: (4)
          - _target_: torchvision.transforms.ToTensor
          - _target_: torchvision.transforms.Normalize
            mean: [0.5, 0.5, 0.5]
            std: [0.5, 0.5, 0.5]

```
</div>
1.  `_target_` is a special reserved keyword that initializes a python object from the provided text. In this case, a `Trainer` object from the `pytorch_lightning` library is initialized
2.  `max_epochs` is an argument of the `Trainer` class which is passed through this format. Any argument for the class can be passed similarly.
3.  `$@` is a combination of `$` which evaluates a python expression and `@` which references a python object. In this case we first reference the model with `@model` which is the `torchvision.models.resnet18` defined earlier and then access its parameters using `$@model.parameters()`
4.  YAML allows passing a list in the format below where each `_target_` specifies a transform that is added to the list of transforms in `Compose`. The `torchvision.datasets.CIFAR10` accepts these with a `transform` argument and applies them to each item.

5.  Datasets are defined for different modes: train, val, test, and predict. Each dataset can have its own transforms and configurations.

## Configuration Concepts
As seen in the [Quickstart](./quickstart.md), Lighter has two main components:

### Trainer Setup
```yaml
    trainer:
        _target_: pytorch_lightning.Trainer # (1)!
        max_epochs: 100
```

The trainer object (`pytorch_lightning.Trainer`) is initialized using the `_target_` key. For more information on `_target_` and other special keys, see [Special Syntax and Keywords](#special-syntax-and-keywords).

The `max_epochs` parameter is passed to the `pytorch_lightning.Trainer` object during instantiation. You can provide any argument accepted by the class in this manner.

### LighterSystem Configuration
While Lighter utilizes the Trainer from PyTorch Lightning, LighterSystem is a unique component that incorporates concepts from PL, such as LightningModule, to encapsulate all essential elements of a deep learning system in a straightforward manner.

Concepts encapsulated by LighterSystem include,

#### Model definition
The `torchvision` library is included by default in Lighter, allowing you to select various torchvision models. Additionally, Lighter includes `monai`, enabling you to easily switch to a ResNet model by adjusting your configuration as follows:

=== "Torchvision ResNet18"

    ```yaml
    LighterSystem:
      ...
      model:
        _target_: torchvision.models.resnet18
        num_classes: 10
      ...
    ```

=== "MONAI ResNet50"

    ```yaml
    LighterSystem:
      ...
      model:
        _target_: monai.networks.nets.resnet50
        num_classes: 10
        spatial_dims: 2
      ...
    ```

=== "MONAI 3DResNet50"

    ```yaml
    LighterSystem:
      ...
      model:
        _target_: monai.networks.nets.resnet50
        num_classes: 10
        spatial_dims: 3 
      ...
    ```

<br/>
#### Criterion/Loss

Just as you can override models, Lighter allows you to switch between various loss functions from libraries like `torch` and `monai`. This flexibility lets you experiment with different optimization strategies without altering your code. Here are examples of how to modify the criterion section in your configuration to use different loss functions:

=== "CrossEntropyLoss"
    ```yaml
    LighterSystem:
      ...
      criterion:
        _target_: torch.nn.CrossEntropyLoss
      ...
    ```

=== "MONAI's Dice Loss"
    ```yaml
    LighterSystem:
      ...
      criterion:
        _target_: monai.losses.DiceLoss
      ...
    ```

<br/>
#### Optimizer

Similarly, you can experiment with different optimizer parameters. Model parameters are passed directly to the optimizer via the `params` argument.
```yaml hl_lines="5" 
LighterSystem:
  ...
  optimizer:
    _target_: torch.optim.Adam
    params: "$@system#model.parameters()"
    lr: 0.001
  ...
```

You can also define a scheduler for the optimizer as shown below:
```yaml hl_lines="10"
LighterSystem:
  ...
  optimizer:
    _target_: torch.optim.Adam
    params: "$@system#model.parameters()"
    lr: 0.001

  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    optimizer: "@system#optimizer"
    eta_min: 1.0e-06
    T_max: "%trainer#max_epochs"

  ...
```
In this example, the optimizer is passed to the scheduler using the `optimizer` argument. The `%trainer#max_epochs` syntax retrieves the `max_epochs` value from the Trainer class.

<br/>
#### Datasets

Datasets are often the most frequently modified part of the configuration, as workflows typically involve training or inferring on custom datasets. The `datasets` key includes `train`, `val`, `test`, and `predict` sub-keys, which generate dataloaders for each workflow supported by PyTorch Lightning. Detailed information is available [here](./workflows.md).

<div class="annotate" markdown>

```yaml
LighterSystem:
  ...
  datasets:
    train:
      _target_: torchvision.datasets.CIFAR10 (1)
      download: True
      root: .datasets
      train: True
      transform: (2)
        _target_: torchvision.transforms.Compose
        transforms:
          - _target_: torchvision.transforms.ToTensor
          - _target_: torchvision.transforms.Normalize
            mean: [0.5, 0.5, 0.5]
            std: [0.5, 0.5, 0.5]
  ...
```

</div>
1. Define your own dataset class here or use existing dataset classes. Learn more about this [here](./projects.md).
2.  Transforms can be applied to each dataset element by initializing a `Compose` object and providing a list of transforms. This approach is often the best way to adapt constraints for your data.

### Special Syntax and Keywords
- `_target_`: Specifies the Python class to instantiate. If a function is provided, a partial function is created. Any configuration key with `_target_` maps to a Python object.
- **@**: References another configuration value. This syntax allows access to keys mapped to Python objects. For example, the learning rate of an optimizer instantiated as `torch.optim.Adam` can be accessed using `@model#lr`, where `lr` is an attribute of the `torch.optim.Adam` class.
- **$**: Evaluates Python expressions.
- **%**: Acts as a macro for textual replacement in the configuration.
