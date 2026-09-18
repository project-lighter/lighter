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

## Usage

```bash
pip install lighter
cd projects/cifar10

# Train
lighter fit configs/example.yaml

# Quick test
lighter fit configs/example.yaml trainer::fast_dev_run=true

# Longer training
lighter fit configs/example.yaml trainer::max_epochs=50
```

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)


## Evaluation protocol

The example reserves 10% of the official training population for validation, selected with the independent fixed `split_seed: 42`. Training and validation indices are disjoint; their union is the official training set. Validation uses evaluation transforms. The official test population is used only by test/predict. Keep the split seed fixed while comparing model or training-seed changes, and select hyperparameters/checkpoints with validation results before opening the test results. These are example split choices, not a framework-imposed dataset policy.

Earlier configurations used the official test set as validation. Results from that protocol are not directly comparable to the corrected holdout protocol. Tiny fixture checks exercise membership and deterministic splitting; full dataset training was not used to certify model quality.
