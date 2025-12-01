# Self-Supervised Learning with SimCLR

Contrastive learning using the [lightly](https://github.com/lightly-ai/lightly) library.

## Dataset

**CIFAR-10** - 60,000 32x32 images. Auto-downloaded.

## Method

**SimCLR** - creates two augmented views per image and uses NT-Xent loss to learn representations by maximizing agreement between views.

## Lighter Features Demonstrated

- **lightly integration** - SSL transforms and losses
- **Model-computed loss** - no criterion config needed
- **Temperature parameter** - for contrastive loss tuning
- **`drop_last: true`** - required for contrastive learning

## Requirements

```bash
pip install lightly
```

## Usage

```bash
cd projects/self_supervised

# Quick test
uv run --project ../.. lighter fit configs/simclr.yaml

# Full training
lighter fit configs/simclr.yaml trainer::fast_dev_run=false
```

## References

- [SimCLR Paper](https://arxiv.org/abs/2002.05709)
- [lightly Documentation](https://docs.lightly.ai/)
