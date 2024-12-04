# Configuration System

Lighter is a configuration-centric framework where the config. is used for setting up the machine learning workflow from model architecture selection, loss function, optimizer, dataset preparation and running the training/evaluation/inference process.

Our configuration system is heavily based on the MONAI bundle parser but with a standardized structure. For every configuration, we expect several items to be mandatorily defined.

The configuration is divided into two main components:
- **Trainer**: Handles the training process, including epochs, devices, etc.
- **LighterSystem**: Encapsulates the model, optimizer, datasets, and other components.

Let us take a simple example config to dig deeper into the configuration system of Lighter. You can go through the config and click on the + for more information about specific concepts.

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

The trainer object (`pytorch_lightning.Trainer`) is initialized through the `_target_` key. For more info on `_target_` and special keys, click [here](#special-syntax-and-keywords)

The `max_epochs` is an argument provided to the `pytorch_lightning.Trainer` object during its instantiation. All arguments that are accepted during instantiation can be provided similarly. 

### LighterSystem Configuration
While Lighter borrows the Trainer from Pytorch Lightning, LighterSystem is a custom component unique to Lighter that draws on several concepts of PL such as LightningModule to provide a simple way to capture all the integral elements of a deep learning system. 

Concepts encapsulated by LighterSystem include,

#### Model definition
The `torchvision` library is installed by default in Lighter and therefore, you can choose different torchvision models here. We also have `monai` packaged with Lighter, so if you are looking to use a ResNet, all you need to modify to fit this new model in your config is,

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

Similar to overriding models, when exploring different loss types in Lighter, you can easily switch between various loss functions provided by libraries such as `torch` and `monai`. This flexibility allows you to experiment with different approaches to optimize your model's performance without changing code!! Below are some examples of how you can modify the criterion section in your configuration file to use different loss functions.

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

Same as above, you can experiment with different optimizer parameters. Model parameters are directly passed to the optimizer in `params` argument.
```yaml hl_lines="5" 
LighterSystem:
  ...
  optimizer:
    _target_: torch.optim.Adam
    params: "$@system#model.parameters()"
    lr: 0.001
  ...
```

 You can also define a scheduler for the optimizer as below,
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
Here, the optimizer is passed to the scheduler with the `optimizer` argument. `%trainer#max_epochs` is also passed to the scheduler where it fetches `max_epochs` from the Trainer class.

<br/>
#### Datasets

The most commonly changed part of the config is often the datasets as common workflows involve training/inferring on your own dataset. We provide a `datasets` key with `train`, `val`, `test` and `predict` keys that generate dataloaders for each of the different workflows provided by pytorch lightning. These are described in detail [here](./workflows.md)

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
1. Define your own dataset class here or use several existing dataset clases. Read more about [this](./projects.md)
2.  Transforms can be applied to each element of the dataset by initialization a `Compose` object and providing it a list of transforms. This is often the best way to adapt constraints for your data. 

### Special Syntax and Keywords
- `_target_`: Indicates the Python class to instantiate. If a function is provided, a partial function is created. Any configuration key set with `_target_` will map to a python object. 
- **@**: References another configuration value. Using this syntax, keys mapped to python objects can be accessed. For instance, the learning rate of an optimizer, `optimizer` instianted to `torch.optim.Adam` using `_target_` can be accessed using `@model#lr` where `lr` is an attribute of the `torch.optim.Adam` class.
- **$**: Used for evaluating Python expressions.
- **%**: Macro for textual replacement in the configuration.
