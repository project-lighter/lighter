---
title: Training Guide
---

# Training Guide

Run experiments, track results, and save outputs with Lighter.

This guide covers the full training workflow from config to results.

## Basic Commands

Lighter provides four main commands:

```bash
# Train and validate
lighter fit config.yaml

# Validate only (requires checkpoint)
lighter validate config.yaml

# Test only (requires checkpoint)
lighter test config.yaml

# Run inference
lighter predict config.yaml
```

All commands use the same config structure.

## The Fit Command

Train your model with automatic validation:

```bash
lighter fit config.yaml
```

### What Happens

1. Loads config from YAML
2. Instantiates trainer, model, and data
3. Runs training loop with validation
4. Saves checkpoints automatically
5. Logs metrics to configured logger

### Output Structure

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── config.yaml          # Copy of config used
        ├── checkpoints/
        │   ├── last.ckpt       # Latest checkpoint
        │   └── epoch=09-step=1000.ckpt
        └── logs/               # Tensorboard/CSV logs
```

### Resuming Training

Resume from latest checkpoint:

```bash
lighter fit config.yaml --ckpt_path path/to/checkpoint.ckpt
```

## Overriding from CLI

Change any config value without editing files:

### Single Override

```bash
# Change learning rate
lighter fit config.yaml model::optimizer::lr=0.01

# Train longer
lighter fit config.yaml trainer::max_epochs=100

# Use more GPUs
lighter fit config.yaml trainer::devices=4
```

### Multiple Overrides

```bash
lighter fit config.yaml \
  model::optimizer::lr=0.01 \
  trainer::max_epochs=100 \
  data::train_dataloader::batch_size=64 \
  trainer::devices=4
```

### Nested Overrides

```bash
# Override nested values
lighter fit config.yaml \
  model::network::num_classes=100 \
  model::optimizer::weight_decay=0.0001
```

### Complex Overrides

```bash
# Add callbacks from CLI
lighter fit config.yaml \
  'trainer::callbacks=[{_target_: pytorch_lightning.callbacks.EarlyStopping, monitor: val/loss}]'
```

## Merging Configs

Combine multiple YAML files:

```bash
lighter fit base.yaml,experiment.yaml
```

### Example: Base + Experiment

`base.yaml`:

```yaml
trainer:
  max_epochs: 100
  accelerator: auto
  devices: 1

model:
  _target_: models.MyModel
  network:
    _target_: torchvision.models.resnet18
    num_classes: 10

data:
  _target_: lighter.LighterDataModule
  train_dataloader:
    batch_size: 32
```

`experiment.yaml`:

```yaml
# Override specific values
trainer:
  max_epochs: 200  # Override
  devices: 4       # Add

model:
  optimizer:
    lr: 0.01  # Add optimizer config
```

**Result**: Merged config with `max_epochs=200`, `devices=4`, new optimizer.

### Merge Operators

Control how configs merge:

**Replace with `=`**:

```yaml
# experiment.yaml
trainer:
  =callbacks:  # Replace entire list
    - _target_: pytorch_lightning.callbacks.EarlyStopping
      monitor: val/loss
```

**Delete with `~`**:

```yaml
# experiment.yaml
trainer:
  ~callbacks: null  # Remove callbacks entirely

data:
  ~test_dataloader: null  # Remove test dataloader
```

## Checkpointing

### Automatic Checkpointing

Lightning saves `last.ckpt` automatically. For more control:

```yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      dirpath: checkpoints
      filename: 'epoch{epoch:02d}-loss{val/loss:.4f}'
      monitor: val/loss
      mode: min
      save_top_k: 3          # Keep best 3
      save_last: true        # Keep last checkpoint
      every_n_epochs: 1      # Save every epoch
```

### Save Based on Metric

```yaml
# Save best validation accuracy
- _target_: pytorch_lightning.callbacks.ModelCheckpoint
  monitor: val/acc
  mode: max
  save_top_k: 1
  filename: 'best-acc{val/acc:.4f}'
```

### Multiple Checkpointers

Save different metrics:

```yaml
trainer:
  callbacks:
    # Best accuracy
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      monitor: val/acc
      mode: max
      save_top_k: 1
      filename: 'best-acc'

    # Best loss
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      monitor: val/loss
      mode: min
      save_top_k: 1
      filename: 'best-loss'

    # Regular saves
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      every_n_epochs: 10
      filename: 'epoch{epoch:02d}'
```

### Loading Checkpoints

**For validation/testing**:

```bash
lighter validate config.yaml --ckpt_path checkpoints/best.ckpt
lighter test config.yaml --ckpt_path checkpoints/best.ckpt
```

**For inference**:

```bash
lighter predict config.yaml --ckpt_path checkpoints/best.ckpt
```

**To resume training**:

```bash
lighter fit config.yaml --ckpt_path checkpoints/last.ckpt
```

## Logging

### TensorBoard (Default)

```yaml
trainer:
  logger:
    _target_: pytorch_lightning.loggers.TensorBoardLogger
    save_dir: logs
    name: my_experiment
```

View logs:

```bash
tensorboard --logdir logs
```

### CSV Logger

```yaml
trainer:
  logger:
    _target_: pytorch_lightning.loggers.CSVLogger
    save_dir: logs
    name: my_experiment
```

Results saved to `logs/my_experiment/version_0/metrics.csv`.

### Weights & Biases

```yaml
trainer:
  logger:
    _target_: pytorch_lightning.loggers.WandbLogger
    project: my_project
    name: experiment_1
    save_dir: logs
```

### Multiple Loggers

Use all at once:

```yaml
trainer:
  logger:
    - _target_: pytorch_lightning.loggers.TensorBoardLogger
      save_dir: logs

    - _target_: pytorch_lightning.loggers.CSVLogger
      save_dir: logs

    - _target_: pytorch_lightning.loggers.WandbLogger
      project: my_project
```

### No Logging

Disable logging:

```yaml
trainer:
  logger: false
```

## Saving Predictions

Use Writers to save predictions to files.

### CSV Writer

Save predictions to CSV:

```yaml
trainer:
  callbacks:
    - _target_: lighter.callbacks.CSVWriter
      write_interval: batch  # or 'epoch'
```

Your `predict_step` should return a dict:

```python
def predict_step(self, batch, batch_idx):
    x, y = batch
    pred = self(x)

    return {
        "prediction": pred.argmax(dim=1),
        "probability": pred.max(dim=1).values,
        "target": y,
    }
```

Output: `predictions.csv` with columns for each key.

### File Writer

Save predictions to individual files:

```yaml
trainer:
  callbacks:
    - _target_: lighter.callbacks.FileWriter
      write_interval: batch
```

Return dict with data and filenames:

```python
def predict_step(self, batch, batch_idx, dataloader_idx=0):
    images, paths = batch

    predictions = self(images)

    # Save each prediction
    results = []
    for i, (pred, path) in enumerate(zip(predictions, paths)):
        results.append({
            "prediction": pred.cpu().numpy(),
            "$id": f"pred_{batch_idx}_{i}",  # Unique filename
        })

    return results
```

Saves: `predictions/pred_0_0.npz`, `pred_0_1.npz`, etc.

### Custom Writer

Create your own:

```python
from lighter.callbacks import BaseWriter

class CustomWriter(BaseWriter):
    def write(self, data):
        """Save data however you want."""
        # data is what you returned from predict_step
        output_path = self.output_dir / f"{data['$id']}.pkl"

        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
```

Use in config:

```yaml
trainer:
  callbacks:
    - _target_: my_project.writers.CustomWriter
      write_interval: batch
```

## Debugging

### Fast Dev Run

Run 1 batch of train/val/test to catch bugs:

```bash
lighter fit config.yaml trainer::fast_dev_run=true
```

Or specify number of batches:

```bash
lighter fit config.yaml trainer::fast_dev_run=5
```

### Overfit on Small Batch

Test if model can overfit (sanity check):

```bash
lighter fit config.yaml trainer::overfit_batches=10
```

Trains on same 10 batches repeatedly.

### Limit Batches

Run partial epoch:

```bash
# Train on 10% of data
lighter fit config.yaml \
  trainer::limit_train_batches=0.1 \
  trainer::limit_val_batches=0.1
```

Or specific number:

```bash
lighter fit config.yaml trainer::limit_train_batches=100
```

### Profiler

Profile your code:

```bash
lighter fit config.yaml trainer::profiler=simple
```

Options:

- `simple` - Basic profiling
- `advanced` - Detailed profiling
- `pytorch` - PyTorch profiler

Results saved to logs directory.

### Find Learning Rate

Automatically find optimal LR:

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  callbacks:
    - _target_: pytorch_lightning.callbacks.LearningRateFinder
      min_lr: 1e-6
      max_lr: 1.0
```

Or run tuner:

```bash
lighter fit config.yaml trainer::auto_lr_find=true
```

## Multi-GPU Training

### Single Machine, Multiple GPUs

```yaml
trainer:
  devices: 4  # Use 4 GPUs
  strategy: ddp  # Distributed Data Parallel
```

Or use all available GPUs:

```yaml
trainer:
  devices: -1  # All GPUs
  strategy: ddp
```

### Strategy Options

**DDP (Recommended)**:

```yaml
trainer:
  strategy: ddp
```

**DDP Spawn**:

```yaml
trainer:
  strategy: ddp_spawn
```

**DeepSpeed**:

```yaml
trainer:
  strategy:
    _target_: pytorch_lightning.strategies.DeepSpeedStrategy
    stage: 2
```

**FSDP (Fully Sharded)**:

```yaml
trainer:
  strategy: fsdp
```

### Batch Size Adjustment

Scale batch size with GPUs:

```yaml
vars:
  num_gpus: 4
  per_gpu_batch: 32

data:
  train_dataloader:
    batch_size: "$%vars::per_gpu_batch * %vars::num_gpus"
```

Or keep per-GPU batch size:

```yaml
# Each GPU gets batch_size=32
data:
  train_dataloader:
    batch_size: 32
```

## Mixed Precision Training

Use 16-bit precision for faster training:

```yaml
trainer:
  precision: 16
```

Or BFloat16:

```yaml
trainer:
  precision: "bf16-mixed"
```

Automatic mixed precision (AMP) is handled by Lightning.

## Gradient Accumulation

Simulate larger batch sizes:

```yaml
trainer:
  accumulate_grad_batches: 4
```

Effective batch size = `batch_size × accumulate_grad_batches`.

Example:

```yaml
# Effective batch size = 32 × 4 = 128
data:
  train_dataloader:
    batch_size: 32

trainer:
  accumulate_grad_batches: 4
```

## Early Stopping

Stop training when metric stops improving:

```yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.EarlyStopping
      monitor: val/loss
      patience: 10
      mode: min
      verbose: true
```

Parameters:

- `monitor`: Metric to track
- `patience`: Epochs to wait before stopping
- `mode`: `min` or `max`
- `min_delta`: Minimum change to qualify as improvement

## Progress Bars

### Default Progress Bar

Shows by default. Disable with:

```yaml
trainer:
  enable_progress_bar: false
```

### Custom Progress Bar

```yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.RichProgressBar
```

Or:

```yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.TQDMProgressBar
      refresh_rate: 10
```

## Validation

### Validate Only

Run validation on a checkpoint:

```bash
lighter validate config.yaml --ckpt_path checkpoints/best.ckpt
```

### Validation Frequency

Validate every N epochs:

```yaml
trainer:
  check_val_every_n_epoch: 5
```

Or every N steps:

```yaml
trainer:
  val_check_interval: 0.5  # Validate twice per epoch
```

Or specific number of steps:

```yaml
trainer:
  val_check_interval: 100  # Every 100 training steps
```

### Skip Validation

```yaml
trainer:
  limit_val_batches: 0  # No validation
```

## Testing

Run final test after training:

```bash
# Fit then test automatically
lighter fit config.yaml

# Test separately
lighter test config.yaml --ckpt_path checkpoints/best.ckpt
```

### Test During Fit

Not recommended, but possible by loading checkpoint at end of fit.

## Prediction/Inference

Run inference on data:

```bash
lighter predict config.yaml --ckpt_path checkpoints/best.ckpt
```

Requires:

1. `predict_step` in your module
2. `predict_dataloader` in your data config
3. Optional: Writer callback to save results

Example config:

```yaml
data:
  predict_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 32
    dataset:
      _target_: my_project.data.PredictionDataset
      root: ./inference_data

trainer:
  callbacks:
    - _target_: lighter.callbacks.FileWriter
      write_interval: batch
```

Example predict_step:

```python
def predict_step(self, batch, batch_idx):
    images = batch
    predictions = self(images)

    return {
        "predictions": predictions.cpu(),
        "batch_idx": batch_idx,
    }
```

## Experiment Organization

### Recommended Structure

```
my_project/
├── __lighter__.py
├── models.py
├── data.py
├── configs/
│   ├── base.yaml           # Baseline config
│   ├── resnet50.yaml       # Architecture variants
│   ├── augmented.yaml      # Augmentation experiments
│   └── ablation/
│       ├── no_dropout.yaml
│       └── no_batchnorm.yaml
└── outputs/                # Generated by Lighter
    └── YYYY-MM-DD/
        └── HH-MM-SS/
```

### Config Naming

Use descriptive names:

```
configs/
├── baseline-resnet18.yaml
├── baseline-resnet50.yaml
├── lr0.01-batch128.yaml
├── augment-strong.yaml
└── finetune-imagenet.yaml
```

### Version Control

Track configs in git:

```bash
git add configs/
git commit -m "Add strong augmentation experiment"
```

Compare experiments:

```bash
git diff configs/baseline.yaml configs/improved.yaml
```

## Common Workflows

### Workflow 1: Hyperparameter Search

Create configs for different hyperparameters:

```bash
# Try different learning rates
lighter fit base.yaml model::optimizer::lr=0.001
lighter fit base.yaml model::optimizer::lr=0.01
lighter fit base.yaml model::optimizer::lr=0.1

# Try different architectures
lighter fit base.yaml model::network::_target_=torchvision.models.resnet18
lighter fit base.yaml model::network::_target_=torchvision.models.resnet50
lighter fit base.yaml model::network::_target_=torchvision.models.efficientnet_b0
```

### Workflow 2: Resume Failed Training

Training crashed? Resume:

```bash
lighter fit config.yaml --ckpt_path outputs/2024-01-15/10-30-45/checkpoints/last.ckpt
```

### Workflow 3: Incremental Training

Train, then finetune:

```bash
# Initial training
lighter fit pretrain.yaml

# Finetune with lower LR
lighter fit finetune.yaml \
  --ckpt_path outputs/.../checkpoints/last.ckpt \
  model::optimizer::lr=0.0001
```

### Workflow 4: Cross-Validation

Run multiple folds:

```bash
for fold in {0..4}; do
  lighter fit config.yaml data::fold=$fold
done
```

Config:

```python
# data.py
class CVDataset(Dataset):
    def __init__(self, root, fold, num_folds=5):
        # Split data by fold
        ...
```

## Output Management

### Change Output Directory

```yaml
# In config
trainer:
  default_root_dir: ./my_outputs
```

Or CLI:

```bash
lighter fit config.yaml trainer::default_root_dir=./my_outputs
```

### Disable Checkpoints

```yaml
trainer:
  enable_checkpointing: false
```

### Save Frequency

Save less often:

```yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      every_n_epochs: 10  # Save every 10 epochs
```

Or based on steps:

```yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      every_n_train_steps: 1000
```

## Troubleshooting

### Out of Memory

**Solutions:**

1. Reduce batch size:
   ```bash
   lighter fit config.yaml data::train_dataloader::batch_size=16
   ```

2. Use gradient accumulation:
   ```yaml
   trainer:
     accumulate_grad_batches: 4
   ```

3. Use mixed precision:
   ```yaml
   trainer:
     precision: 16
   ```

4. Reduce model size:
   ```bash
   lighter fit config.yaml model::network::_target_=torchvision.models.resnet18
   ```

### Training Too Slow

**Solutions:**

1. Use more workers:
   ```yaml
   data:
     train_dataloader:
       num_workers: 8
   ```

2. Pin memory:
   ```yaml
   data:
     train_dataloader:
       pin_memory: true
   ```

3. Use multiple GPUs:
   ```yaml
   trainer:
     devices: 4
     strategy: ddp
   ```

4. Mixed precision:
   ```yaml
   trainer:
     precision: 16
   ```

### Model Not Learning

**Debug steps:**

1. Overfit on small batch:
   ```bash
   lighter fit config.yaml trainer::overfit_batches=10
   ```

2. Check learning rate:
   ```bash
   lighter fit config.yaml trainer::auto_lr_find=true
   ```

3. Visualize data:
   ```python
   # In training_step
   if batch_idx == 0:
       self.logger.experiment.add_images("train/batch", x[:8])
   ```

4. Profile:
   ```bash
   lighter fit config.yaml trainer::profiler=simple
   ```

## Next Steps

- [Best Practices](best-practices.md) - Production patterns
- [Example Projects](../examples/index.md) - Complete working examples
- [CLI Reference](../reference/cli.md) - Full command documentation

## Quick Reference

```bash
# Basic commands
lighter fit config.yaml
lighter validate config.yaml
lighter test config.yaml
lighter predict config.yaml

# Override from CLI
lighter fit config.yaml key::path=value

# Merge configs
lighter fit base.yaml,experiment.yaml

# Resume training
lighter fit config.yaml --ckpt_path path/to/last.ckpt

# Multi-GPU
lighter fit config.yaml trainer::devices=4 trainer::strategy=ddp

# Debug
lighter fit config.yaml trainer::fast_dev_run=true
lighter fit config.yaml trainer::overfit_batches=10
```
