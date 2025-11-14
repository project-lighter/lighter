# Running Experiments

Lighter streamlines deep learning experiments through well-defined stages. This guide covers everything from basic runs to advanced workflows.

## Stages Overview

Lighter orchestrates experiments through four distinct stages, each serving a specific purpose:

| Stage | Purpose | Common Use Cases |
|-------|---------|------------------|
| **`fit`** | Train model on training data | Initial training, fine-tuning, transfer learning |
| **`validate`** | Evaluate on validation data | Hyperparameter tuning, model selection |
| **`test`** | Final evaluation on test data | Performance benchmarking, final metrics |
| **`predict`** | Generate predictions on new data | Inference, deployment, data analysis |

## Quick Start

### Basic Commands

```bash
# Train a model
lighter fit config.yaml

# Validate a trained model
lighter validate config.yaml

# Test final performance
lighter test config.yaml

# Generate predictions
lighter predict config.yaml
```

### Common Workflows

#### 1. Full Training Pipeline
```bash
# Train + validate (automatic validation during training)
lighter fit config.yaml

# Then test on held-out data - equivalent to Trainer.test(..., ckpt_path="best")
lighter test config.yaml args::test::ckpt_path="best.ckpt"
```

#### 2. Resume Training from Checkpoint
```bash
lighter fit config.yaml args::fit::ckpt_path="last.ckpt"
```

#### 3. Fine-tuning Pre-trained Model
```bash
# Use base config + fine-tuning overrides
lighter fit base_config.yaml,finetune_config.yaml
```

## Advanced Configuration

### Stage-Specific Arguments

The `args` key provides a way to pass arguments directly to the PyTorch Lightning `Trainer`'s stage methods: `fit`, `validate`, `test`, and `predict`. Each key under `args` corresponds to a stage, and the arguments within it are passed to that stage's method. For instance, `args.test.ckpt_path` is passed as `Trainer.test(ckpt_path=...)`.

Configure stage arguments in your YAML or via CLI:

```yaml title="config.yaml"
args:
    fit:
        ckpt_path: null  # Start from scratch
    validate:
        ckpt_path: "checkpoints/best.ckpt"
    test:
        ckpt_path: "checkpoints/best.ckpt"
    predict:
        ckpt_path: "checkpoints/best.ckpt"
        return_predictions: true
```

### CLI Override Patterns

```bash
# Override any config parameter
lighter fit config.yaml trainer::max_epochs=50

# Override nested parameters
lighter fit config.yaml system::optimizer::lr=0.001

# Override list elements
lighter fit config.yaml trainer::callbacks::0::patience=10

# Multiple overrides
lighter fit config.yaml \
    trainer::max_epochs=100 \
    system::optimizer::lr=0.0001 \
    trainer::devices=2
```

## Pro Tips 💡

### 1. Config Composition
Combine multiple configs for modular experiments:
```bash
# Base + dataset + model + training configs
lighter fit base.yaml,data/cifar10.yaml,models/resnet.yaml,train/sgd.yaml
```

### 2. Quick Experimentation
Test configurations without full training:
```bash
# Fast run with 2 batches
lighter fit config.yaml trainer::fast_dev_run=2
```

### 3. GPU Management
```bash
# Use specific GPUs
lighter fit config.yaml trainer::devices=[0,1]

# Use all available GPUs
lighter fit config.yaml trainer::devices=-1
```

### 4. Debugging Runs
```bash
# Enable detailed logging
lighter fit config.yaml trainer::log_every_n_steps=1

# Profile your code
lighter fit config.yaml trainer::profiler="simple"
```

## Common Patterns

### Pattern 1: Hyperparameter Search
```bash
# Run multiple experiments with different learning rates
for lr in 0.001 0.01 0.1; do
    lighter fit config.yaml \
        system::optimizer::lr=$lr \
        trainer::logger::name="lr_$lr"
done
```

### Pattern 2: Cross-Validation
```bash
# Run k-fold cross-validation
for fold in {1..5}; do
    lighter fit config.yaml \
        system::dataloaders::train::dataset::fold=$fold \
        trainer::logger::name="fold_$fold"
done
```

### Pattern 3: Progressive Training
```bash
# Start with small resolution, then increase
lighter fit config.yaml vars::image_size=128 trainer::max_epochs=10
lighter fit config.yaml vars::image_size=256 args::fit::ckpt_path="last.ckpt"
lighter fit config.yaml vars::image_size=512 args::fit::ckpt_path="last.ckpt"
```

## Troubleshooting

### Issue: Out of Memory
```bash
# Reduce batch size
lighter fit config.yaml system::dataloaders::train::batch_size=8

# Enable gradient accumulation
lighter fit config.yaml trainer::accumulate_grad_batches=4

# Use mixed precision
lighter fit config.yaml trainer::precision="16-mixed"
```

### Issue: Training Too Slow
```bash
# Increase number of workers
lighter fit config.yaml system::dataloaders::train::num_workers=8

# Enable compile mode (PyTorch 2.0+)
lighter fit config.yaml system::model::compile=true
```

### Issue: Validation Takes Too Long
```bash
# Reduce validation frequency
lighter fit config.yaml trainer::check_val_every_n_epoch=5

# Limit validation batches
lighter fit config.yaml trainer::limit_val_batches=0.25
```

## Environment Variables

Control Lighter behavior with environment variables:

```bash
# Set random seed for reproducibility via Pytorch Lightning
PL_GLOBAL_SEED=42 lighter fit config.yaml

# Enable debugging mode
LIGHTER_DEBUG=1 lighter fit config.yaml
```

## Quick Reference

| Task | Command |
|------|---------|
| Train from scratch | `lighter fit config.yaml` |
| Resume training | `lighter fit config.yaml args::fit::ckpt_path="last.ckpt"` |
| Validate checkpoint | `lighter validate config.yaml args::validate::ckpt_path="best.ckpt"` |
| Test model | `lighter test config.yaml args::test::ckpt_path="best.ckpt"` |
| Generate predictions | `lighter predict config.yaml args::predict::ckpt_path="best.ckpt"` |
| Fast debugging | `lighter fit config.yaml trainer::fast_dev_run=true` |
| Multi-GPU training | `lighter fit config.yaml trainer::devices=4` |
| Mixed precision | `lighter fit config.yaml trainer::precision="16-mixed"` |

## Recap and Next Steps

You now have a comprehensive understanding of running experiments with Lighter. Key takeaways:

* Four stages (`fit`, `validate`, `test`, `predict`) cover the full ML lifecycle
* Flexible configuration through YAML files and CLI overrides
* Powerful composition and workflow patterns
* Built-in solutions for common issues

## Related Guides
- [Configuration Guide](configuration.md) - Config syntax and patterns
- [Troubleshooting](troubleshooting.md) - Common errors and solutions
- [Experiment Tracking](experiment_tracking.md) - Logging experiments
