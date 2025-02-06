# End-to-End Semantic Segmentation

## Introduction

Semantic segmentation is key in computer vision, classifying each pixel into categories. This tutorial guides you to train a semantic segmentation model using Lighter, focusing on medical imaging data and U-Net. We'll cover dataset loading, model definition, config, training, prediction, and visualization.

## Dataset: Medical Segmentation Decathlon (MSD) - Liver

We use the Liver segmentation task from MSD dataset. MSD: 10 datasets for medical image segmentation. Liver dataset: CT scans, manual liver segmentations. MONAI will be used for easy data access/processing.

### Loading MSD Liver Dataset with Lighter

To load the MSD Liver dataset, we will configure the `dataloaders` section in `config.yaml` using MONAI's dataset and dataloader classes.

```yaml title="config.yaml"
system:
  dataloaders:
    train:
      _target_: monai.data.DataLoader
      dataset:
        _target_: monai.data.Dataset
        data:
          _target_: monai.apps.load_and_split_data
          dataset_name: "MSD_Liver"
          data_dir: ".datasets/"
          is_Thor=False # CT images
          val_frac: 0.2
          seed: 42
          section: "training" # "training" or "validation"
        transform:
          _target_: monai.transforms.Compose
          transforms:
            - _target_: monai.transforms.LoadImaged
              keys: ["image", "label"] # "image" and "label" fields in input dict
            - _target_: monai.transforms.AddChanneld
              keys: ["image", "label"]
            - _target_: monai.transforms.Orientationd
              keys: ["image", "label"]
              axcodes: "RAS" # Standard medical image orientation
            - _target_: monai.transforms.Spacingd
              keys: ["image", "label"]
              pixdim: [1.5, 1.5, 2.0] # Adjust spacing as needed
              mode: ["bilinear", "nearest"] # Interpolation modes
            - _target_: monai.transforms.ScaleIntensityRanged
              keys: ["image"]
              a_min: -57
              a_max: 164
              b_min: 0.0
              b_max: 1.0
              clip=True
            - _target_: monai.transforms.CropForegroundd
              keys: ["image", "label"]
              source_key: "image" # Use image to determine foreground
            - _target_: monai.transforms.RandCropByPosNegLabeld
              keys: ["image", "label"]
              label_key: "label"
              spatial_size: [96, 96, 96] # Patch size
              pos=1
              neg=1
              num_samples=4 # Number of patches per image
              image_key="image"
              image_threshold=0 # Threshold to determine valid patches
            - _target_: monai.transforms.RandAffined
              keys: ["image", "label"]
              mode= ["bilinear", "nearest"]
              prob=0.5
              spatial_dims=3
              rotate_range=null
              shear_range=null
              translate_range=null
              scale_range=null
            - _target_: monai.transforms.ToTensord
              keys: ["image", "label"]
      batch_size: 2 # Adjust based on GPU memory
      shuffle: true
      num_workers: 2 # Adjust based on system resources
```

Let's break down dataloader config:

*   **`_target_: monai.data.DataLoader`**: MONAI `DataLoader`.
*   **`dataset`**: MONAI `Dataset`.
    *   **`data`**: Data loading/splitting via `monai.apps.load_and_split_data`.
        *   **`dataset_name: "MSD_Liver"`**: MSD Liver dataset.
        *   **`data_dir: ".datasets/"`**: Dataset dir.
        *   **`is_Thor=False`**: CT images (not MRI).
        *   **`val_frac: 0.2`**: Validation split fraction.
        *   **`seed: 42`**: Random seed.
        *   **`section: "training"`**: Load training split.
    *   **`transform`**: Data transforms via `monai.transforms.Compose`.
        *   **`LoadImaged`**: Load image/label files.
        *   **`AddChanneld`**: Add channel dim (3D images).
        *   **`Orientationd`**: Standardize orientation (RAS).
        *   **`Spacingd`**: Resample to specified spacing.
        *   **`ScaleIntensityRanged`**: Scale intensity to [0, 1].
        *   **`CropForegroundd`**: Crop foreground.
        *   **`RandCropByPosNegLabeld`**: Random patch extraction (balanced labels).
        *   **`RandAffined`**: Random affine augmentation.
        *   **`ToTensord`**: To PyTorch tensors.
*   **`batch_size: 2`**: Training batch size.
*   **`shuffle: true`**: Shuffle data.
*   **`num_workers: 2`**: Data loader workers (adjust for system).

You can define a validation dataloader similarly, changing `section: "validation"` in the `load_and_split_data` configuration.

## Model: U-Net

We use classic U-Net for semantic segmentation. MONAI provides easy U-Net integration.

### Defining U-Net in `config.yaml`

Configure model in `system.model` section of `config.yaml`:

```yaml title="config.yaml"
system:
  model:
    _target_: monai.networks.nets.UNet
    dimensions: 3 # 3D images
    in_channels: 1 # Input channels (CT)
    out_channels: 2 # Output channels (liver/background)
    channels: [16, 32, 64, 128, 256] # Feature channels per layer
    dropout: 0.0 # Dropout probability
    num_res_units: 2 # Residual units per layer
```

Config uses `monai.networks.nets.UNet`, sets params for 3D Liver segmentation U-Net:

*   **`dimensions: 3`**: 3D U-Net.
*   **`in_channels: 1`**: 1 input channel (CT scans).
*   **`out_channels: 2`**: 2 output channels (liver & background).
*   **`channels`**: Feature channels per U-Net layer.
*   **`dropout: 0.0`**: No dropout.
*   **`num_res_units: 2`**: Residual units per layer.

## Loss Function and Inferer

Dice loss is common for semantic segmentation. We'll use MONAI's `SlidingWindowInferer` for efficient 3D medical image inference.

### Configuring Loss and Inferer in `config.yaml`

Add to `system` config in `config.yaml`:

```yaml title="config.yaml"
system:
  criterion:
    _target_: monai.losses.DiceLoss
    to_onehot_y: true # Convert labels to one-hot
    softmax: true # Apply softmax to predictions

  inferer:
    _target_: monai.inferers.SlidingWindowInferer
    roi_size: [96, 96, 96] # Sliding window ROI size
    sw_batch_size: 4 # Sliding window batch size
    overlap: 0.5 # Sliding window overlap
```

*   **`criterion`**: Dice loss via `monai.losses.DiceLoss`.
    *   **`to_onehot_y: true`**: Convert labels to one-hot.
    *   **`softmax: true`**: Apply softmax to predictions.
*   **`inferer`**: `SlidingWindowInferer` setup.
    *   **`roi_size: [96, 96, 96]`**: ROI (patch) size for sliding window inference.
    *   **`sw_batch_size: 4`**: Batch size for sliding window patches.
    *   **`overlap: 0.5`**: Sliding window overlap ratio.

## Metrics and Logging

Dice metric evaluates segmentation. `FileWriter` callback saves validation predictions.

### Configuring Metrics and FileWriter in `config.yaml`

Add to `system.metrics` and `trainer.callbacks` in `config.yaml`:

```yaml title="config.yaml"
system:
  metrics:
    val:
      - _target_: monai.metrics.DiceMetric
        include_background: false # Exclude background class
        reduction: "mean_batch" # Average metric over batch
    test: # Test stage metrics
      - _target_: monai.metrics.DiceMetric
        include_background: false
        reduction: "mean_batch"

trainer:
  callbacks:
    - _target_: lighter.callbacks.FileWriter
      path: "outputs/predictions" # Prediction save directory
      writer: "itk_seg_nrrd" # ITK NRRD segmentation writer
```

*   **`metrics`**: Dice metric for validation/test via `monai.metrics.DiceMetric`:
    *   **`include_background: false`**: Exclude background class (index 0).
    *   **`reduction: "mean_batch"`**: Average Dice score over batch.
*   **`callbacks`**: `FileWriter` callback config:
    *   **`_target_: lighter.callbacks.FileWriter`**: `FileWriter` callback.
    *   **`path: "outputs/predictions"`**: Prediction output directory.
    *   **`writer: "itk_seg_nrrd"`**: `itk_seg_nrrd` writer (ITK NRRD format).

## Complete Configuration (`config.yaml`)

Complete `config.yaml` for training 3D U-Net for liver segmentation on MSD Liver dataset:

```yaml title="config.yaml"
trainer:
  accelerator: "auto" # Use GPU if available, else CPU
  max_epochs: 20 # Train for 20 epochs
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint # Save best checkpoint
      monitor: "val/metrics/DiceMetric/epoch" # Monitor validation Dice metric
      mode: "max" # Save when metric maximized
      filename: "best_model" # Checkpoint file name prefix
    - _target_: lighter.callbacks.FileWriter
      path: "outputs/predictions"
      writer: "itk_seg_nrrd"

system:
  _target_: lighter.System

  model:
    _target_: monai.networks.nets.UNet
    dimensions: 3 # 3D images
    in_channels: 1 # Input channels (CT)
    out_channels: 2 # Output channels (liver/background)
    channels: [16, 32, 64, 128, 256]
    dropout: 0.0
    num_res_units: 2

  criterion:
    _target_: monai.losses.DiceLoss
    to_onehot_y: true
    softmax: true

  inferer:
    _target_: monai.inferers.SlidingWindowInferer
    roi_size: [96, 96, 96]
    sw_batch_size: 4
    overlap: 0.5

  metrics:
    val:
      - _target_: monai.metrics.DiceMetric
        include_background: false
        reduction: "mean_batch"
    test:
      - _target_: monai.metrics.DiceMetric
        include_background: false
        reduction: "mean_batch"

  dataloaders:
    train:
      _target_: monai.data.DataLoader
      dataset:
        _target_: monai.data.Dataset
        data:
          _target_: monai.apps.load_and_split_data
          dataset_name: "MSD_Liver"
          data_dir: ".datasets/"
          is_Thor=False
          val_frac: 0.2
          seed: 42
          section: "training"
        transform:
          _target_: monai.transforms.Compose
          transforms:
            - _target_: monai.transforms.LoadImaged
              keys: ["image", "label"]
            - _target_: monai.transforms.AddChanneld
              keys: ["image", "label"]
            - _target_: monai.transforms.Orientationd
              keys: ["image", "label"]
              axcodes: "RAS"
            - _target_: monai.transforms.Spacingd
              keys: ["image", "label"]
              pixdim: [1.5, 1.5, 2.0]
              mode: ["bilinear", "nearest"]
            - _target_: monai.transforms.ScaleIntensityRanged
              keys: ["image"]
              a_min: -57
              a_max: 164
              b_min: 0.0
              b_max: 1.0
              clip=True
            - _target_: monai.transforms.CropForegroundd
              keys: ["image", "label"]
              source_key: "image"
            - _target_: monai.transforms.RandCropByPosNegLabeld
              keys: ["image", "label"]
              label_key: "label"
              spatial_size: [96, 96, 96]
              pos=1
              neg=1
              num_samples=4
              image_key="image"
              image_threshold=0
            - _target_: monai.transforms.RandAffined
              keys: ["image", "label"]
              mode= ["bilinear", "nearest"]
              prob=0.5
              spatial_dims=3
              rotate_range=null
              shear_range=null
              translate_range=null
              scale_range=null
            - _target_: monai.transforms.ToTensord
              keys: ["image", "label"]
    val:
      _target_: monai.data.DataLoader
      dataset:
        _target_: monai.data.Dataset
        data:
          _target_: monai.apps.load_and_split_data
          dataset_name: "MSD_Liver"
          data_dir: ".datasets/"
          is_Thor=False
          val_frac: 0.2
          seed: 42
          section: "validation" # Use validation split
        transform: # Use same transforms as training, or adjust
          _target_: monai.transforms.Compose
          transforms:
            - _target_: monai.transforms.LoadImaged
              keys: ["image", "label"]
            - _target_: monai.transforms.AddChanneld
              keys: ["image", "label"]
            - _target_: monai.transforms.Orientationd
              keys: ["image", "label"]
              axcodes: "RAS"
            - _target_: monai.transforms.Spacingd
              keys: ["image", "label"]
              pixdim: [1.5, 1.5, 2.0]
              mode: ["bilinear", "nearest"]
            - _target_: monai.transforms.ScaleIntensityRanged
              keys: ["image"]
              a_min: -57
              a_max: 164
              b_min: 0.0
              b_max: 1.0
              clip=True
            - _target_: monai.transforms.CropForegroundd
              keys: ["image", "label"]
              source_key: "image"
            - _target_: monai.transforms.RandCropByPosNegLabeld # Optional, remove or adjust for validation
              keys: ["image", "label"]
              label_key: "label"
              spatial_size: [96, 96, 96]
              pos=1
              neg=1
              num_samples=4
              image_key="image"
              image_threshold=0
            - _target_: monai.transforms.ToTensord
              keys: ["image", "label"]
    test: # Optional test dataloader
      _target_: monai.data.DataLoader
      dataset:
        _target_: monai.data.Dataset
        data:
          _target_: monai.apps.load_and_split_data
          dataset_name: "MSD_Liver"
          data_dir: ".datasets/"
          is_Thor=False
          val_frac: 0.0 # No validation split, use full test set
          seed: 42
          section: "test" # Load test set
        transform: # Use same transforms as training, or adjust
          _target_: monai.transforms.Compose
          transforms:
            - _target_: monai.transforms.LoadImaged
              keys: ["image", "label"]
            - _target_: monai.transforms.AddChanneld
              keys: ["image", "label"]
            - _target_: monai.transforms.Orientationd
              keys: ["image", "label"]
              axcodes: "RAS"
            - _target_: monai.transforms.Spacingd
              keys: ["image", "label"]
              pixdim: [1.5, 1.5, 2.0]
              mode: ["bilinear", "nearest"]
            - _target_: monai.transforms.ScaleIntensityRanged
              keys: ["image"]
              a_min: -57
              a_max: 164
              b_min: 0.0
              b_max: 1.0
              clip=True
            - _target_: monai.transforms.CropForegroundd
              keys: ["image", "label"]
              source_key: "image" # Use image to determine foreground
            - _target_: monai.transforms.ToTensord
              keys: ["image", "label"]
optimizer:
    _target_: torch.optim.AdamW
    lr: 1.0e-4
    weight_decay: 1.0e-5
scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    T_max: "$@trainer.max_epochs" # Cosine Annealing based on max epochs
---
This complete configuration includes:

*   **`trainer`**: Configures PyTorch Lightning Trainer for 20 epochs, saves the best model checkpoint based on validation Dice metric, and sets up the `FileWriter` callback.
*   **`system`**: Defines the Lighter System with:
    *   **`model`**: 3D U-Net from MONAI.
    *   **`criterion`**: Dice loss.
    *   **`inferer`**: `SlidingWindowInferer`.
    *   **`metrics`**: Dice metric for validation and test.
    *   **`dataloaders`**: DataLoaders for train, val, and test stages using MSD Liver dataset and MONAI transforms.
*   **`optimizer`**: AdamW optimizer.
*   **`scheduler`**: Cosine Annealing learning rate scheduler.

## Training and Prediction

To start training, save the above configuration as `config_segmentation.yaml`. Then, run:

```bash title="Terminal"
lighter fit --config config_segmentation.yaml
```

Monitor the training progress in your terminal. After training, you can run prediction on the validation or test set using:

```bash title="Terminal"
lighter predict --config config_segmentation.yaml
```

The `FileWriter` callback will save the segmentation predictions in the `outputs/predictions` directory in ITK NRRD format.

## Visualization

To visualize the saved segmentation outputs, you can use medical image viewers like [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php). ITK-SNAP supports the NRRD format and allows you to overlay segmentations on the original CT images to visually assess the results.

## Recap and Next Steps

In this tutorial, you have learned how to train a 3D U-Net model for semantic segmentation of the liver using Lighter and MONAI. You covered:

*   Loading and preprocessing a medical image segmentation dataset (MSD Liver) using MONAI.
*   Configuring a 3D U-Net model, Dice loss, and `SlidingWindowInferer` in `config.yaml`.
*   Setting up metrics and a `FileWriter` callback for evaluation and saving predictions.
*   Running training and prediction using the Lighter CLI.
*   Visualizing segmentation results with ITK-SNAP.

This tutorial provides a starting point for tackling more complex medical image segmentation tasks with Lighter. In the next tutorials, we will explore [transfer learning](04_transfer_learning.md) and [multi-task learning](05_multi_task_learning.md). You can also explore the [How-To guides](../how-to/01_custom_project_modules.md) for more specific tasks and customizations, and the [Design section](../design/01_overview.md) for deeper insights into Lighter's design principles.
