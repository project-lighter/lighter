# CIFAR-10 Image Classification

Reference example - simple CNN for CIFAR-10 classification.

## Dataset

**CIFAR-10** - 60,000 32x32 color images, 10 classes. Auto-downloaded.

## Architecture

3-layer CNN (~62K parameters): conv layers with ReLU/pooling, 2 FC layers.

## Lighter Features Demonstrated

- **`$` expressions** - `$@model::network.parameters()`
- **`%` raw references** - `%::train_metrics` for config reuse
- **MetricCollection** - Accuracy, F1, Precision, Recall
- **FileWriter callback** - saves predictions as tensors

## Requirements

No extra dependencies.

## Usage

```bash
cd projects/cifar10

# Quick test
uv run --project ../.. lighter fit configs/example.yaml

# Full training
lighter fit configs/example.yaml trainer::max_epochs=50
```

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
