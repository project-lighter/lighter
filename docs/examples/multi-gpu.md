---
title: Multi-GPU Training
---

# Multi-GPU Training

Scale your training across multiple GPUs with Distributed Data Parallel (DDP).

This guide shows how to train on multiple GPUs using PyTorch Lightning's DDP strategy through Lighter configs.

## Quick Start

Train on all available GPUs:

```bash
lighter fit config.yaml trainer::devices=-1 trainer::strategy=ddp
```

Train on specific number of GPUs:

```bash
lighter fit config.yaml trainer::devices=4 trainer::strategy=ddp
```

That's it! Your code works unchanged.

## Configuration

### In Config File

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  devices: 4  # Use 4 GPUs
  strategy: ddp  # Distributed Data Parallel
  accelerator: auto  # Automatically use CUDA if available
```

### From CLI

Override devices:

```bash
# All GPUs
lighter fit config.yaml trainer::devices=-1

# Specific GPUs (0, 1, 2, 3)
lighter fit config.yaml trainer::devices=4

# Specific GPU IDs
lighter fit config.yaml 'trainer::devices=[0,2,3]'
```

## DDP Strategy

### Basic DDP

Recommended for most cases:

```yaml
trainer:
  strategy: ddp
```

Features:
- Each GPU gets own process
- Gradients synchronized across GPUs
- Model replicated on each GPU
- Data split across GPUs

### DDP Spawn

Alternative that spawns subprocesses:

```yaml
trainer:
  strategy: ddp_spawn
```

Use when:
- DDP doesn't work on your system
- Debugging (easier to see errors)

**Note:** Slightly slower than DDP.

### DDP Find Unused Parameters

If you get "unused parameters" error:

```yaml
trainer:
  strategy:
    _target_: pytorch_lightning.strategies.DDPStrategy
    find_unused_parameters: true
```

## Batch Size Considerations

### Per-GPU Batch Size

Each GPU processes `batch_size` samples:

```yaml
data:
  train_dataloader:
    batch_size: 32  # Each GPU: 32 samples
```

**Effective batch size** with 4 GPUs = 32 × 4 = 128

### Keep Total Batch Size

To keep same total batch size across different GPU counts:

```yaml
vars:
  num_gpus: 4
  total_batch_size: 128

data:
  train_dataloader:
    batch_size: "$%vars::total_batch_size // %vars::num_gpus"
```

Override for different GPU counts:

```bash
# 1 GPU: batch_size = 128
lighter fit config.yaml vars::num_gpus=1

# 4 GPUs: batch_size = 32 per GPU
lighter fit config.yaml vars::num_gpus=4

# 8 GPUs: batch_size = 16 per GPU
lighter fit config.yaml vars::num_gpus=8
```

## Learning Rate Scaling

### Linear Scaling Rule

When increasing batch size, scale LR proportionally:

```yaml
vars:
  num_gpus: 4
  base_lr: 0.001

model:
  optimizer:
    lr: "$%vars::base_lr * %vars::num_gpus"
```

**Example:**
- 1 GPU: LR = 0.001, batch = 32
- 4 GPUs: LR = 0.004, batch = 128 (32×4)

### Square Root Scaling

Alternative for very large batch sizes:

```yaml
model:
  optimizer:
    lr: "$%vars::base_lr * (%vars::num_gpus ** 0.5)"
```

## Complete Multi-GPU Example

`experiments/multi_gpu.yaml`:

```yaml
vars:
  # Hardware
  num_gpus: 4

  # Dataset
  num_classes: 10

  # Hyperparameters
  base_lr: 0.001
  total_batch_size: 512  # Total across all GPUs
  max_epochs: 100

  # Computed
  per_gpu_batch_size: "$%vars::total_batch_size // %vars::num_gpus"
  scaled_lr: "$%vars::base_lr * %vars::num_gpus"

trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: "%vars::max_epochs"
  devices: "%vars::num_gpus"
  strategy: ddp
  accelerator: auto

  # Recommended settings
  sync_batchnorm: true  # Sync batch normalization
  precision: 16  # Mixed precision

  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      monitor: val/acc
      mode: max
      save_top_k: 3

    - _target_: pytorch_lightning.callbacks.LearningRateMonitor
      logging_interval: epoch

  logger:
    _target_: pytorch_lightning.loggers.TensorBoardLogger
    save_dir: logs
    name: multi_gpu_experiment

model:
  _target_: lighter.LighterModule

  network:
    _target_: torchvision.models.resnet50
    num_classes: "%vars::num_classes"

  criterion:
    _target_: torch.nn.CrossEntropyLoss

  optimizer:
    _target_: torch.optim.AdamW
    params: "$@model::network.parameters()"
    lr: "%vars::scaled_lr"
    weight_decay: 0.0001

  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    optimizer: "@model::optimizer"
    T_max: "%vars::max_epochs"

  train_metrics:
    - _target_: torchmetrics.Accuracy
      task: multiclass
      num_classes: "%vars::num_classes"

  val_metrics: "%model::train_metrics"

data:
  _target_: lighter.LighterDataModule

  train_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: "%vars::per_gpu_batch_size"
    shuffle: true
    num_workers: 8  # Increase for multi-GPU
    pin_memory: true
    persistent_workers: true
    dataset:
      _target_: torchvision.datasets.CIFAR10
      root: ./data
      train: true
      download: true
      transform:
        _target_: torchvision.transforms.Compose
        transforms:
          - _target_: torchvision.transforms.RandomCrop
            size: 32
            padding: 4
          - _target_: torchvision.transforms.RandomHorizontalFlip
          - _target_: torchvision.transforms.ToTensor
          - _target_: torchvision.transforms.Normalize
            mean: [0.4914, 0.4822, 0.4465]
            std: [0.2470, 0.2435, 0.2616]

  val_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: "%vars::per_gpu_batch_size"
    num_workers: 8
    pin_memory: true
    dataset:
      _target_: torchvision.datasets.CIFAR10
      root: ./data
      train: false
      transform:
        _target_: torchvision.transforms.Compose
        transforms:
          - _target_: torchvision.transforms.ToTensor
          - _target_: torchvision.transforms.Normalize
            mean: [0.4914, 0.4822, 0.4465]
            std: [0.2470, 0.2435, 0.2616]
```

Run:

```bash
lighter fit experiments/multi_gpu.yaml
```

## Advanced Strategies

### FSDP (Fully Sharded Data Parallel)

For very large models that don't fit on single GPU:

```yaml
trainer:
  strategy: fsdp
  devices: 4
```

FSDP shards:
- Model parameters
- Gradients
- Optimizer states

Across GPUs, saving memory.

### DeepSpeed

For even larger models:

```yaml
trainer:
  strategy:
    _target_: pytorch_lightning.strategies.DeepSpeedStrategy
    stage: 2  # ZeRO Stage 2
  devices: 4
  precision: 16
```

Stages:
- **Stage 1**: Shard optimizer states
- **Stage 2**: Shard gradients + Stage 1
- **Stage 3**: Shard parameters + Stage 2

### DDP with Static Graph

For maximum performance (PyTorch 1.11+):

```yaml
trainer:
  strategy:
    _target_: pytorch_lightning.strategies.DDPStrategy
    static_graph: true
```

**Requirements:**
- Model structure doesn't change between steps
- No dynamic control flow

**Benefit:** ~10% speedup.

## Data Loading Optimization

### Increase num_workers

More workers for multi-GPU:

```yaml
data:
  train_dataloader:
    num_workers: "$%vars::num_gpus * 4"  # 4 workers per GPU
```

### Use Persistent Workers

Avoid worker respawning:

```yaml
data:
  train_dataloader:
    persistent_workers: true
```

### Pin Memory

Faster GPU transfer:

```yaml
data:
  train_dataloader:
    pin_memory: true
```

## Gradient Accumulation

Simulate even larger batch sizes:

```yaml
trainer:
  accumulate_grad_batches: 4
```

**Effective batch size** = `batch_size × num_gpus × accumulate_grad_batches`

Example with 4 GPUs:
- `batch_size = 32`
- `num_gpus = 4`
- `accumulate_grad_batches = 4`
- **Effective = 32 × 4 × 4 = 512**

## Sync Batch Normalization

Important for small per-GPU batch sizes:

```yaml
trainer:
  sync_batchnorm: true
```

Synchronizes batch norm statistics across GPUs.

**Use when:** Per-GPU batch size < 8.

## Mixed Precision

Combine with multi-GPU for maximum speed:

```yaml
trainer:
  precision: 16  # or "bf16-mixed"
  devices: 4
  strategy: ddp
```

**Speedup:** ~2-3× faster than FP32.

## Monitoring Multi-GPU Training

### TensorBoard

Same as single GPU:

```bash
tensorboard --logdir logs
```

Metrics automatically aggregated across GPUs.

### Weights & Biases

Works out of the box:

```yaml
trainer:
  logger:
    _target_: pytorch_lightning.loggers.WandbLogger
    project: my_project
```

Only rank 0 process logs to avoid duplicates.

## Checkpointing

Same as single GPU:

```yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      save_top_k: 3
```

Only rank 0 saves checkpoints automatically.

## Testing Locally

Test DDP on single machine with multiple physical GPUs:

```bash
# Run DDP on 2 physical GPUs
lighter fit config.yaml trainer::devices=2 trainer::strategy=ddp
```

**Important:** `devices=k` selects `k` physical GPUs per node (equivalent to `list(range(k))`). No GPU virtualization or simulation is performed. You must have at least as many physical GPUs as specified (e.g., at least 2 physical GPUs for the example above).

## Common Issues

### Out of Memory

**Solutions:**

1. Reduce per-GPU batch size:
   ```bash
   lighter fit config.yaml data::train_dataloader::batch_size=16
   ```

2. Use gradient accumulation:
   ```yaml
   trainer:
     accumulate_grad_batches: 2
   ```

3. Use FSDP or DeepSpeed for large models

### Slow Startup

**Problem:** Long startup time with DDP

**Cause:** Dataset download or preprocessing on each rank

**Solution:** Download data before training:

```bash
# Download once
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', download=True)"

# Then train
lighter fit config.yaml
```

### Hanging at Initialization

**Problem:** Process hangs at "Initializing distributed"

**Solutions:**

1. Check firewall settings
2. Try different DDP backend:
   ```yaml
   trainer:
     strategy:
       _target_: pytorch_lightning.strategies.DDPStrategy
       process_group_backend: gloo  # Instead of nccl
   ```

### Different Results Across GPUs

**Problem:** Metrics differ between runs

**Cause:** Random seed not set or data shuffling

**Solution:**

```python
# In __lighter__.py
import pytorch_lightning as pl
pl.seed_everything(42, workers=True)
```

```yaml
data:
  train_dataloader:
    shuffle: true  # Ensure shuffling
```

### Unused Parameters Error

**Problem:** "RuntimeError: Expected to have finished reduction in the prior iteration"

**Solution:**

```yaml
trainer:
  strategy:
    _target_: pytorch_lightning.strategies.DDPStrategy
    find_unused_parameters: true
```

## Performance Tips

### 1. Use All CPU Cores

```yaml
data:
  train_dataloader:
    num_workers: "$%vars::num_gpus * 4"
```

### 2. Prefetch Data

```yaml
data:
  train_dataloader:
    prefetch_factor: 2
```

### 3. Mixed Precision

```yaml
trainer:
  precision: 16
```

### 4. Compile Model (PyTorch 2.0+)

In your module:

```python
def __init__(self, network, ...):
    super().__init__()
    self.network = torch.compile(network)
```

### 5. Optimize Data Loading

- Cache dataset if it fits in RAM
- Preprocess data offline
- Use fast storage (SSD > HDD)

## Scaling Example

Compare different GPU counts:

```bash
# 1 GPU baseline
lighter fit config.yaml vars::num_gpus=1

# 2 GPUs (~1.8× speedup)
lighter fit config.yaml vars::num_gpus=2

# 4 GPUs (~3.5× speedup)
lighter fit config.yaml vars::num_gpus=4

# 8 GPUs (~6.5× speedup)
lighter fit config.yaml vars::num_gpus=8
```

**Expected scaling:** ~85-90% efficiency (linear would be 100%).

## Multi-Node Training

For training across multiple machines:

```yaml
trainer:
  strategy: ddp
  devices: 4  # GPUs per node
  num_nodes: 2  # Number of machines
```

Run on each node:

```bash
# Node 0
MASTER_ADDR=node0_address MASTER_PORT=12345 \
  lighter fit config.yaml \
  trainer::num_nodes=2 \
  trainer::devices=4

# Node 1
MASTER_ADDR=node0_address MASTER_PORT=12345 NODE_RANK=1 \
  lighter fit config.yaml \
  trainer::num_nodes=2 \
  trainer::devices=4
```

Requires:
- Shared filesystem for checkpoints
- Network connectivity between nodes
- Matching software environment

## Quick Reference

```yaml
# Basic multi-GPU
trainer:
  devices: 4
  strategy: ddp

# All GPUs
trainer:
  devices: -1
  strategy: ddp

# Large models
trainer:
  strategy: fsdp
  devices: 4

# Very large models
trainer:
  strategy:
    _target_: pytorch_lightning.strategies.DeepSpeedStrategy
    stage: 3
  devices: 4

# Batch size scaling
vars:
  num_gpus: 4
  total_batch: 512

data:
  train_dataloader:
    batch_size: "$%vars::total_batch // %vars::num_gpus"

# LR scaling
model:
  optimizer:
    lr: "$%vars::base_lr * %vars::num_gpus"
```

## Next Steps

- [Training Guide](../guides/training.md) - More training strategies
- [Best Practices](../guides/best-practices.md) - Production optimization
- [Image Classification Example](image-classification.md) - Complete example

## Summary

Multi-GPU training with Lighter:

- ✅ Simple config changes only
- ✅ Code works unchanged
- ✅ Automatic gradient synchronization
- ✅ Linear scaling with proper settings
- ✅ Multiple strategies (DDP, FSDP, DeepSpeed)
- ✅ Works with all Lightning features

**Key takeaway:** Add `trainer::devices=4 trainer::strategy=ddp` to use 4 GPUs. That's it!
