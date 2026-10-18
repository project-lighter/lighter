# CIFAR-10 Image Classification

Reference example - simple CNN for CIFAR-10 classification.

## Dataset

**CIFAR-10** - 60,000 32x32 color images, 10 classes. Auto-downloaded. Validation uses 5,000 of the training
images, held out by `project.dataset.cifar10_split` with a fixed `split_seed`, so the 10,000 test images are used
only by `lighter test` and `lighter predict`.

## Architecture

3-layer CNN (~62K parameters): conv layers with ReLU/pooling, 2 FC layers.

## Lighter Features Demonstrated

- **Optimizer as a partial** - `_mode_: callable`, called by LighterModule with the model's parameters
- **`%` raw references** - `%::train_metrics` for config reuse
- **MetricCollection** - Accuracy, F1, Precision, Recall
- **FileWriter callback** - saves predictions as tensors

## Usage

Needs `torchvision` besides Lighter.

```bash
pip install lighter torchvision
cd projects/cifar10

# Check every stage on random images, without a download (CI runs this)
lighter fit configs/example.yaml configs/fake_data.yaml

# Train
lighter fit configs/example.yaml

# Quick test
lighter fit configs/example.yaml trainer::fast_dev_run=true

# Longer training
lighter fit configs/example.yaml trainer::max_epochs=50
```

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
