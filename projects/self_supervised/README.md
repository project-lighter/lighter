# Self-Supervised Learning with SimCLR

Contrastive learning using the [lightly](https://github.com/lightly-ai/lightly) library.

## Overview

Self-supervised learning (SSL) learns representations from unlabeled data. This project implements **SimCLR** (Simple Framework for Contrastive Learning), which learns by maximizing agreement between differently augmented views of the same image.

## How SimCLR Works

1. **Two views**: Each image is augmented twice to create two different views
2. **Encoding**: Both views pass through the same encoder network
3. **Projection**: Features are projected to a lower-dimensional space
4. **Contrastive loss**: NT-Xent loss pulls together views of the same image while pushing apart views of different images

## Dataset

**CIFAR-10** - 60,000 32x32 images. Auto-downloaded.

The lightly library applies SimCLR-specific augmentations:
- Random resized crop
- Color jitter (brightness, contrast, saturation, hue)
- Random grayscale
- Gaussian blur
- Random horizontal flip

## Architecture

- **Backbone**: ResNet-18 (configurable to ResNet-50)
- **Projection head**: MLP with hidden layer (2048) → output (128)
- **Loss**: NT-Xent (Normalized Temperature-scaled Cross Entropy)

## Lighter Features Demonstrated

- **lightly integration** - SSL transforms, losses, and dataset wrappers
- **Model-computed loss** - No `criterion` config needed (loss computed internally)
- **Custom `__init__`** - Extending LighterModule with additional parameters
- **Temperature parameter** - Configurable via `vars::` section
- **`drop_last: true`** - Required for contrastive learning (batch size consistency)

## Usage

```bash
pip install lighter lightly
cd projects/self_supervised

# Train (100 epochs)
lighter fit configs/simclr.yaml

# Quick test
lighter fit configs/simclr.yaml trainer::fast_dev_run=true

# Different backbone
lighter fit configs/simclr.yaml model::network::backbone=resnet50

# Adjust temperature
lighter fit configs/simclr.yaml vars::temperature=0.1

# Larger batch size (important for SSL performance)
lighter fit configs/simclr.yaml vars::batch_size=512
```

## Configuration Highlights

```yaml
vars:
  batch_size: 256      # Larger is better for contrastive learning
  temperature: 0.5     # Lower = harder negatives
  projection_dim: 128  # Output dimension of projection head

model:
  _target_: project.models.SimCLRModel
  temperature: "%vars::temperature"
  # No criterion - loss is computed internally

data:
  train_dataloader:
    drop_last: true  # Required for contrastive learning
```

## Downstream Evaluation

After pretraining, extract features for downstream tasks:

```bash
lighter predict configs/simclr.yaml --ckpt_path path/to/checkpoint.ckpt
```

The `predict_step` returns features (not projections) suitable for:
- **Linear probing**: Train a linear classifier on frozen features
- **Fine-tuning**: Use pretrained backbone with task-specific head

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `batch_size` | 256 | Larger batches improve performance significantly |
| `temperature` | 0.5 | Lower values focus on hard negatives |
| `projection_dim` | 128 | Output dimension of projection head |
| `hidden_dim` | 2048 | Hidden layer size in projection head |
| Learning rate | 0.06 | Scales with batch size (0.3 * batch_size / 256) |

## Alternative Methods

The `networks/encoder.py` also includes a BYOL (Bootstrap Your Own Latent) implementation which doesn't require negative pairs. To use BYOL, you would need to create a corresponding model class.

## References

- [SimCLR Paper](https://arxiv.org/abs/2002.05709) - Chen et al., 2020
- [lightly Documentation](https://docs.lightly.ai/)
- [SimCLR v2 Paper](https://arxiv.org/abs/2006.10029) - Improved version with larger models
