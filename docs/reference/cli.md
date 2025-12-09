---
title: CLI Reference
---

# CLI Reference

Complete reference for Lighter's command-line interface.

## Commands

Lighter provides four main commands:

```bash
lighter fit        # Train and validate
lighter validate   # Validate only
lighter test       # Test only
lighter predict    # Run inference
```

All commands use the same configuration system.

## lighter fit

Train your model with automatic validation.

### Basic Usage

```bash
lighter fit CONFIG [OPTIONS] [OVERRIDES...]
```

### Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `CONFIG` | Path to YAML config file | Yes |
| `--ckpt_path PATH` | Checkpoint path to resume from ("last", "best", or file path) | No |
| `--weights_only BOOL` | Load only weights (security option) | No |
| `OVERRIDES` | Config overrides (key::path=value) | No |

### Examples

```bash
# Basic training
lighter fit config.yaml

# Resume from checkpoint
lighter fit config.yaml --ckpt_path checkpoints/last.ckpt

# With config overrides
lighter fit config.yaml model::optimizer::lr=0.01

# Multiple configs
lighter fit base.yaml,experiment.yaml

# Combine CLI flags and overrides
lighter fit config.yaml --ckpt_path last trainer::max_epochs=100
```

### Config Structure

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  # ... trainer args ...

model:
  _target_: your.Module
  # ... model args ...

data:
  _target_: lighter.LighterDataModule
  # ... data args ...
```

### Output

Creates directory structure:

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── config.yaml          # Config used
        ├── checkpoints/
        │   └── last.ckpt       # Latest checkpoint
        └── logs/               # Training logs
```

## lighter validate

Run validation on a trained model.

### Basic Usage

```bash
lighter validate CONFIG [OPTIONS] [OVERRIDES...]
```

### Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `CONFIG` | Path to YAML config file | Yes |
| `--ckpt_path PATH` | Checkpoint path for validation ("last", "best", or file path) | No |
| `--verbose BOOL` | Print validation results (default: True) | No |
| `--weights_only BOOL` | Load only weights (security option) | No |
| `OVERRIDES` | Config overrides | No |

### Examples

```bash
# Validate with checkpoint
lighter validate config.yaml --ckpt_path checkpoints/best.ckpt

# Override config
lighter validate config.yaml \
  --ckpt_path checkpoints/best.ckpt \
  data::val_dataloader::batch_size=128
```

### Config Structure

Same as `fit` command.

### Requirements

- Checkpoint file (`.ckpt`)
- `val_dataloader` in data config
- `validation_step` in your module

## lighter test

Run test on a trained model.

### Basic Usage

```bash
lighter test CONFIG [OPTIONS] [OVERRIDES...]
```

### Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `CONFIG` | Path to YAML config file | Yes |
| `--ckpt_path PATH` | Checkpoint path for testing ("last", "best", or file path) | No |
| `--verbose BOOL` | Print test results (default: True) | No |
| `--weights_only BOOL` | Load only weights (security option) | No |
| `OVERRIDES` | Config overrides | No |

### Examples

```bash
# Test with checkpoint
lighter test config.yaml --ckpt_path checkpoints/best.ckpt

# Multiple test sets
lighter test config.yaml \
  --ckpt_path checkpoints/best.ckpt \
  data::test_dataloader::dataset::root=./test_data
```

### Config Structure

Same as `fit` command, with a test dataloader:

```yaml
data:
  test_dataloader:
    _target_: torch.utils.data.DataLoader
    # ... test dataloader config ...
```

### Requirements

- Checkpoint file (`.ckpt`)
- `test_dataloader` in data config
- `test_step` in your module

## lighter predict

Run inference on data.

### Basic Usage

```bash
lighter predict CONFIG [OPTIONS] [OVERRIDES...]
```

### Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `CONFIG` | Path to YAML config file | Yes |
| `--ckpt_path PATH` | Checkpoint path for predictions ("last", "best", or file path) | No |
| `--return_predictions BOOL` | Whether to return predictions (default: True) | No |
| `--weights_only BOOL` | Load only weights (security option) | No |
| `OVERRIDES` | Config overrides | No |

### Examples

```bash
# Basic prediction
lighter predict config.yaml --ckpt_path checkpoints/best.ckpt

# With writer to save results
lighter predict config.yaml \
  --ckpt_path checkpoints/best.ckpt \
  'trainer::callbacks=[{_target_: lighter.callbacks.CSVWriter}]'
```

### Config Structure

Same as `fit` command, with a predict dataloader:

```yaml
data:
  predict_dataloader:
    _target_: torch.utils.data.DataLoader
    # ... predict dataloader config ...

trainer:
  callbacks:
    - _target_: lighter.callbacks.CSVWriter  # Optional: save predictions
      write_interval: batch
```

### Requirements

- Checkpoint file (`.ckpt`)
- `predict_dataloader` in data config
- `predict_step` in your module
- Optional: Writer callback to save results

## Config Overrides

Override any config value from command line.

### Syntax

```bash
lighter COMMAND config.yaml key::path=value
```

### Examples

#### Simple Values

```bash
# Numbers
lighter fit config.yaml model::optimizer::lr=0.01

# Strings
lighter fit config.yaml trainer::logger::name=my_experiment

# Booleans
lighter fit config.yaml trainer::enable_checkpointing=false
```

#### Nested Values

```bash
# Deep nesting
lighter fit config.yaml \
  model::optimizer::lr=0.01 \
  model::optimizer::weight_decay=0.0001 \
  model::network::num_classes=100
```

#### Lists

```bash
# Python list syntax
lighter fit config.yaml 'trainer::devices=[0,1,2,3]'
```

#### Objects

```bash
# YAML object syntax
lighter fit config.yaml \
  'trainer::callbacks=[{_target_: pytorch_lightning.callbacks.EarlyStopping, monitor: val/loss, patience: 10}]'
```

### Path Syntax

Use `::` to navigate config hierarchy:

```yaml
# Config structure
model:
  optimizer:
    lr: 0.001
```

```bash
# Override
lighter fit config.yaml model::optimizer::lr=0.01
```

## Config Merging

Combine multiple config files.

### Syntax

```bash
lighter COMMAND config1.yaml,config2.yaml,...
```

### Behavior

Later files override earlier ones (dictionary merge).

### Examples

```bash
# Base + experiment
lighter fit base.yaml,experiment.yaml

# Multiple overrides
lighter fit base.yaml,data.yaml,model.yaml,overrides.yaml
```

### Example Files

**base.yaml**:
```yaml
trainer:
  max_epochs: 100
  accelerator: auto

model:
  network:
    num_classes: 10
```

**experiment.yaml**:
```yaml
trainer:
  max_epochs: 200  # Override

model:
  optimizer:  # Add
    lr: 0.01
```

**Result**: Merged config with `max_epochs=200` and new optimizer.

## Environment Variables

### LIGHTER_OUTPUT_DIR

Change default output directory:

```bash
export LIGHTER_OUTPUT_DIR=./my_outputs
lighter fit config.yaml
```

Or in config:

```yaml
trainer:
  default_root_dir: ./my_outputs
```

### CUDA_VISIBLE_DEVICES

Control GPU visibility:

```bash
# Use only GPU 2
CUDA_VISIBLE_DEVICES=2 lighter fit config.yaml

# Use GPUs 0 and 3
CUDA_VISIBLE_DEVICES=0,3 lighter fit config.yaml trainer::devices=2
```

### MASTER_ADDR / MASTER_PORT

For multi-node training:

```bash
MASTER_ADDR=node0 MASTER_PORT=12345 lighter fit config.yaml
```

## Common Patterns

### Quick Debugging

```bash
# Fast dev run (1 batch)
lighter fit config.yaml trainer::fast_dev_run=true

# Overfit 10 batches
lighter fit config.yaml trainer::overfit_batches=10

# Limit batches
lighter fit config.yaml trainer::limit_train_batches=0.1
```

### Hyperparameter Tuning

```bash
# Learning rate sweep
for lr in 0.0001 0.001 0.01; do
  lighter fit config.yaml model::optimizer::lr=$lr
done

# Batch size sweep
for bs in 32 64 128 256; do
  lighter fit config.yaml data::train_dataloader::batch_size=$bs
done
```

### Multi-GPU

```bash
# All GPUs
lighter fit config.yaml trainer::devices=-1 trainer::strategy=ddp

# Specific GPUs
lighter fit config.yaml trainer::devices=4 trainer::strategy=ddp

# Specific GPU IDs
lighter fit config.yaml 'trainer::devices=[0,2,3]' trainer::strategy=ddp
```

### Resume Training

```bash
# Resume from last checkpoint
lighter fit config.yaml --ckpt_path outputs/.../checkpoints/last.ckpt

# Resume with different LR
lighter fit config.yaml \
  --ckpt_path outputs/.../checkpoints/last.ckpt \
  model::optimizer::lr=0.0001
```

### Save Predictions

```bash
# CSV output
lighter predict config.yaml \
  --ckpt_path checkpoints/best.ckpt \
  'trainer::callbacks=[{_target_: lighter.callbacks.CSVWriter}]'

# File output
lighter predict config.yaml \
  --ckpt_path checkpoints/best.ckpt \
  'trainer::callbacks=[{_target_: lighter.callbacks.FileWriter}]'
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Config error (invalid YAML, missing files) |
| 2 | Runtime error (training failed, OOM, etc.) |

## Verbosity

PyTorch Lightning controls logging verbosity.

### Reduce Logging

```yaml
trainer:
  enable_progress_bar: false
  enable_model_summary: false
```

### Increase Logging

```bash
# Python logging
export PYTHONWARNINGS=default
lighter fit config.yaml
```

## Tips

### Shell Completion

Add to your shell config:

```bash
# Bash
eval "$(_LIGHTER_COMPLETE=bash_source lighter)"

# Zsh
eval "$(_LIGHTER_COMPLETE=zsh_source lighter)"
```

### Config Validation

Validate config without training:

```bash
lighter fit config.yaml trainer::fast_dev_run=true
```

### Find Config Issues

Enable Sparkwheel debug output:

```bash
SPARKWHEEL_DEBUG=1 lighter fit config.yaml
```

### See Resolved Config

Add print in your module:

```python
def __init__(self, ...):
    super().__init__()
    self.save_hyperparameters()
    print(self.hparams)  # See final values
```

## Next Steps

- [Configuration Guide](../guides/configuration.md) - Learn config syntax
- [Training Guide](../guides/training.md) - Training workflows
- [Example Projects](../examples/index.md) - Complete examples

## Quick Reference

```bash
# Basic commands
lighter fit config.yaml
lighter validate config.yaml
lighter test config.yaml
lighter predict config.yaml

# Overrides
lighter fit config.yaml key::path=value

# Multiple configs
lighter fit base.yaml,experiment.yaml

# Checkpoints
lighter fit config.yaml --ckpt_path path/to/checkpoint.ckpt
lighter validate config.yaml --ckpt_path path/to/checkpoint.ckpt
lighter test config.yaml --ckpt_path path/to/checkpoint.ckpt
lighter predict config.yaml --ckpt_path path/to/checkpoint.ckpt

# Multi-GPU
lighter fit config.yaml trainer::devices=4 trainer::strategy=ddp

# Debugging
lighter fit config.yaml trainer::fast_dev_run=true
```
