# Transfer Learning: Finetuning a Pre-trained Model

## Introduction

Transfer learning is a powerful technique in machine learning that allows you to leverage knowledge gained from solving one problem and apply it to a different but related problem. In deep learning, this often involves using pre-trained models, which are models trained on large datasets (e.g., ImageNet). Finetuning a pre-trained model on a new, smaller dataset can significantly reduce training time and improve performance, especially when data is limited.
  
This tutorial will guide you through finetuning a pre-trained image classification model using Lighter. We will use a ResNet-18 model pre-trained on ImageNet and finetune it on the CIFAR-10 dataset. We will also demonstrate how to use the `Freezer` callback in Lighter to freeze and unfreeze layers during training.

## Pre-trained Model: ResNet-18 on ImageNet

We will use a ResNet-18 model pre-trained on the [ImageNet](http://www.image-net.org/) dataset, a massive dataset of millions of labeled images. PyTorch `torchvision.models` provides easy access to pre-trained models.

### Defining the Pre-trained Model in `config.yaml`

Configure the pre-trained ResNet-18 model in the `system.model` section of your `config.yaml`:

```yaml title="config.yaml"
system:
  model:
    _target_: torchvision.models.resnet18
    pretrained: true # Load pre-trained weights from ImageNet
    num_classes: 10 # Modify the final layer for CIFAR-10 (10 classes)
```

*   **`_target_: torchvision.models.resnet18`**: Specifies ResNet-18 model from `torchvision.models`.
*   **`pretrained: true`**:  Downloads and loads pre-trained weights trained on ImageNet.
*   **`num_classes: 10`**: Modifies the final fully connected layer of ResNet-18 to output 10 classes, matching the number of classes in CIFAR-10.

By setting `pretrained: true`, PyTorch automatically downloads the pre-trained weights. Setting `num_classes` modifies the last fully connected layer to match the new task.

### Loading Pre-trained Weights from Checkpoint

For more control over loading pre-trained weights, e.g. loading from a local checkpoint file with potentially different layer names, use `adjust_prefix_and_load_state_dict` from `lighter.utils.model.py`. 

Example: load weights from `checkpoints/resnet18_imagenet.ckpt` for ResNet-18:

```yaml title="config.yaml"
system:
  model:
    _target_: torchvision.models.resnet18
    _partial_: true # Indicate partial config
    _init_: # Call __init__ after partial config
      _target_: lighter.utils.model.adjust_prefix_and_load_state_dict
      ckpt_path: "checkpoints/resnet18_imagenet.ckpt" # Checkpoint file path
      layers_to_ignore: # Ignore final layer(s)
        - "fc.weight"
        - "fc.bias"
```

Config breakdown:

*   `_partial_: true`, `_init_`: Apply `adjust_prefix_and_load_state_dict` after partial model init.
*   `_init_` calls `lighter.utils.model.adjust_prefix_and_load_state_dict` with:
    *   **`ckpt_path`**: Path to downloaded checkpoint file.
    *   **`layers_to_ignore`**: Layers to exclude from loading. Example ignores final FC layer (`"fc"`).

This shows how to use `adjust_prefix_and_load_state_dict` for flexible pre-trained weight loading, handling different checkpoint sources/architectures and layer mismatches.
 
## Dataset: CIFAR-10

We will use the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, a standard dataset for image classification. CIFAR-10 consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. We will use `torchvision.datasets.CIFAR10` to load this dataset.

### Loading CIFAR-10 Dataset

Configure the CIFAR-10 dataset in the `system.dataloaders` section of your `config.yaml`, similar to the [Image Classification tutorial](02_image_classification.md):

```yaml title="config.yaml"
system:
  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: ".datasets/" # Directory to download dataset
        train: true # Use training split
        download: true # Download if not exists
        transform: # Data transformations
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.RandomResizedCrop
              size: 224 # Resize images to 224x224 for ResNet-18
              scale: [0.8, 1.0]
              ratio: [0.75, 1.3333]
            - _target_: torchvision.transforms.RandomHorizontalFlip
              p: 0.5
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.485, 0.456, 0.406] # ImageNet normalization
              std: [0.229, 0.224, 0.225] # ImageNet normalization
      batch_size: 32
      shuffle: true
      num_workers: 4
    val: # Validation dataloader
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: ".datasets/"
        train: false # Use validation split
        download: false # Already downloaded
        transform: # Validation transforms (no augmentation)
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.Resize
              size: 256 # Resize for validation
            - _target_: torchvision.transforms.CenterCrop
              size: 224 # Center crop to 224x224
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.485, 0.456, 0.406] # ImageNet normalization
              std: [0.229, 0.224, 0.225] # ImageNet normalization
      batch_size: 32
      num_workers: 4

## Optimizer and Learning Rate Scheduler

We will use AdamW optimizer and Cosine Annealing LR scheduler, like previous tutorials.

### Configuring Optimizer and Scheduler

Add optimizer/scheduler configs to `config.yaml`:

```yaml title="config.yaml"
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4 # Lower LR for finetuning
  weight_decay: 1.0e-5

scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: "$@trainer.max_epochs"
```

Note lower learning rate (`lr: 1.0e-4`) vs training from scratch. Finetuning needs smaller LR to avoid disrupting pre-trained weights.

## Freezer Callback: Freezing Layers

To demonstrate freezing, we use `Freezer` callback to freeze ResNet-18 conv layers for initial epochs, then unfreeze.

### Configuring Freezer Callback

Add `Freezer` callback to `trainer.callbacks` in `config.yaml`:

```yaml title="config.yaml"
trainer:
  max_epochs: 20
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      monitor: "val/acc/epoch" # Monitor validation accuracy
      mode: "max"
      filename: "best_model"
    - _target_: lighter.callbacks.Freezer # Freezer callback
      name_starts_with: ["model.backbone"] # Freeze backbone initially
      until_epoch: 5                     # Unfreeze backbone after epoch 5
```

*   **`_target_: lighter.callbacks.Freezer`**: `Freezer` callback.
*   **`name_starts_with`**: Freeze layers starting with prefixes (ResNet-18 conv layers).
*   **`until_epoch: 5`**: Unfreeze layers after epoch 5.

Config freezes layers for first 5 epochs, then trains all layers.

## Metrics

We use accuracy as evaluation metric.

### Configuring Metrics

Add accuracy metric to `system.metrics` section:

```yaml title="config.yaml"
system:
  metrics:
    train: # Metrics for training
      - _target_: torchmetrics.Accuracy
        task: "multiclass"
        num_classes: 10
    val: # Metrics for validation
      - _target_: torchmetrics.Accuracy
        task: "multiclass"
        num_classes: 10
    test: # Metrics for test
      - _target_: torchmetrics.Accuracy
        task: "multiclass"
        num_classes: 10
```

Config configures `torchmetrics.Accuracy` for train/val/test stages (multi-class, 10 classes).

## Complete Configuration (`config.yaml`)

Complete `config.yaml` for finetuning pre-trained ResNet-18 on CIFAR-10:

```yaml title="config.yaml"
trainer:
  accelerator: "auto"
  max_epochs: 20
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      monitor: "val/acc/epoch"
      mode: "max"
      filename: "best_model"
    - _target_: lighter.callbacks.Freezer
      name_starts_with: ["conv1", "layer1", "layer2", "layer3"]
      until_epoch: 5

system:
  _target_: lighter.System

  model:
    _target_: torchvision.models.resnet18
    pretrained: true
    num_classes: 10

  criterion:
    _target_: torch.nn.CrossEntropyLoss

  metrics:
    train:
      - _target_: torchmetrics.Accuracy
        task: "multiclass"
        num_classes: 10
    val:
      - _target_: torchmetrics.Accuracy
        task: "multiclass"
        num_classes: 10
    test:
      - _target_: torchmetrics.Accuracy
        task: "multiclass"
        num_classes: 10

  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: ".datasets/"
        train: true
        download: true
        transform:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.RandomResizedCrop
              size: 224
              scale: [0.8, 1.0]
              ratio: [0.75, 1.3333]
            - _target_: torchvision.transforms.RandomHorizontalFlip
              p: 0.5
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
      batch_size: 32
      shuffle: true
      num_workers: 4
    val:
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: ".datasets/"
        train: false
        download: false
        transform:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.Resize
              size: 256
            - _target_: torchvision.transforms.CenterCrop
              size: 224
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
      batch_size: 32
      num_workers: 4
    test: # Optional test dataloader (same as validation)
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: ".datasets/"
        train: false
        download: false
        transform:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.Resize
              size: 256
            - _target_: torchvision.transforms.CenterCrop
              size: 224
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
      batch_size: 32
      num_workers: 4


optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  weight_decay: 1.0e-5

scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: "$@trainer.max_epochs"
---
Config defines finetuning components: trainer, system, optimizer, scheduler, Freezer callback.

## Training and Evaluation

Save config as `config_transfer_learning.yaml`. Run training:

```bash title="Terminal"
lighter fit --config config_transfer_learning.yaml
```

Monitor training. Validation accuracy should increase quickly due to transfer learning. Freezer callback freezes initial layers for 5 epochs, then trains all.

Evaluate finetuned model on test set (or validation set):

```bash title="Terminal"
lighter test --config config_transfer_learning.yaml
```

Evaluates best checkpoint, reports test accuracy.

## Recap and Next Steps

Tutorial: transfer learning, finetuned pre-trained ResNet-18 on CIFAR-10. Covered:

*   Loading pre-trained ResNet-18 (`torchvision.models`).
*   Configuring CIFAR-10 dataset/dataloaders.
*   Setting up optimizer, LR scheduler, accuracy metric.
*   `Freezer` callback for layer freezing.
*   CLI training/evaluation.

Demonstrates transfer learning effectiveness, Lighter's finetuning simplification. Next: [multi-task learning](05_multi_task_learning.md), [How-To guides](../how-to/01_custom_project_modules.md), [Design section](../design/01_overview.md), [Tutorials section](../tutorials/01_configuration_basics.md).
