## Setting up the config

The config must contain two components - `system` and `trainer`.    

### `system`

`system` encapsulates all parts of a deep learning setup, such as the model, optimizer, criterion, datasets, metrics, etc. `LighterSystem` not only does that, but also implements training logic that is suitable for any task - classification, segmentation, object detection, self-supervised learning, etc.

```yaml
system:
  _target_: lighter.LighterSystem
  batch_size: 512

  model: torchvision.models.resnet18
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

### `trainer`

`trainer` is the component that contains all the information about the training process, such as the number of epochs, the number of gpus, the number of nodes, etc. We rely on the `pytorch_lightning.Trainer` class for this component. Please refer to the [Pytorch Lightning's Trainer documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html) for more information.

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 100
```

## Running the experiment with Lighter
If you have saved the above config as `5-minute-example.yaml`, you can run it as simple as:

```bash
lighter fit --config_file 5-minute-example.yaml
```

## Overriding the config from CLI
We can override any part of the config from the CLI. For example, if we want to change the number of epochs to 200, we can do:
```bash
lighter fit --config_file 5-minute-example.yaml --trainer#max_epochs 200
```

If you're interested to learn more about how config works, please refer to the [MONAI Bundle Configuration documentation](https://docs.monai.io/en/stable/config_syntax.html).