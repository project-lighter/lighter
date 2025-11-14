---
title: System Internals
---

# System Internals

The `System` class extends PyTorch Lightning's `LightningModule` and orchestrates your entire training pipeline. Understanding its operation helps with debugging and customization.

## Overview

System manages:

- Model architecture
- Optimizer and scheduler
- Loss function (criterion)
- Metrics computation
- Data loading
- Adapters for data transformation
- Inference strategies (inferer)

## The Unified `_step()` Method

All modes (train, val, test, predict) use the same `_step()` method:

```
1. Batch → BatchAdapter → [input, target, identifier]
2. Model.forward(input) → prediction
   (or Inferer in val/test/predict modes)
3. CriterionAdapter → Criterion → loss (train/val only)
4. MetricsAdapter → Metrics → values (train/val/test)
5. LoggingAdapter → Logger
6. Output dict → callbacks
```

This unified approach ensures consistency while allowing mode-specific behavior through adapters.

## Automatic Pruning

The Runner automatically removes unused components based on stage:

```yaml
system:
  dataloaders:
    train: ...   # Removed for TEST, PREDICT
    val: ...     # Removed for TEST, PREDICT
    test: ...    # Removed for FIT, VALIDATE, PREDICT
    predict: ... # Removed for FIT, VALIDATE, TEST

  optimizer: ... # Removed for VALIDATE, TEST, PREDICT
  criterion: ... # Removed for TEST, PREDICT
```

This enables **one config for all stages**.

## Mode-Specific Behavior

### Loss Calculation

Loss is calculated only in **train** and **val** modes:

```python
if self.mode in [Mode.TRAIN, Mode.VAL]:
    loss = adapters.criterion(self.criterion, input, target, pred)
```

Test and predict modes return `None`.

### Dict-Based Losses

For multi-task learning, return a dict with `"total"` key:

```python
def my_criterion(pred, target):
    return {
        "total": loss1 + loss2,  # Required for backprop
        "classification": loss1,
        "segmentation": loss2,
    }
```

All sublosses logged automatically; `"total"` used for gradients.

### Metrics Calculation

Metrics calculated in **train**, **val**, and **test** modes (not predict):

```python
if self.mode == Mode.PREDICT or self.metrics[self.mode] is None:
    return None
```

## Special Features

### Epoch/Step Injection

If your model accepts `epoch` or `step` parameters, they're injected automatically:

```python
class MyModel(nn.Module):
    def forward(self, x, epoch=None, step=None):
        # Use for curriculum learning
        if epoch is not None:
            difficulty = min(epoch / self.max_epochs, 1.0)
            x = self.apply_difficulty(x, difficulty)
        return self.process(x)
```

No configuration needed—works automatically.

### Inferer in Val/Test/Predict

In validation, testing, and prediction modes, an inferer can replace the forward pass:

```python
if self.inferer and self.mode in [Mode.VAL, Mode.TEST, Mode.PREDICT]:
    return self.inferer(input, self.model, **kwargs)
return self.model(input, **kwargs)
```

Use for:

- Sliding window inference
- Test-time augmentation
- Ensemble methods
- Custom post-processing

## Automatic Logging

System logs automatically:

### Loss

- Step and epoch level: `{mode}/loss/step`, `{mode}/loss/epoch`
- Individual sublosses for dict-based losses

### Metrics

- Step and epoch level: `{mode}/metrics/{name}/step`, `{mode}/metrics/{name}/epoch`

### Optimizer Stats

Once per epoch during training:

- Learning rate: `train/lr`
- Momentum (SGD)
- Beta values (Adam/AdamW)

## Output Dictionary

Each step returns:

```python
{
    "identifier": batch_identifier,  # Optional
    "input": input_data,              # After LoggingAdapter
    "target": target_data,            # After LoggingAdapter
    "pred": predictions,              # After LoggingAdapter
    "loss": loss_value,               # None in test/predict
    "metrics": metrics_dict,          # None in predict
    "step": self.global_step,
    "epoch": self.current_epoch,
}
```

This dictionary is passed to callbacks for custom processing.

## Customization

Extend System for advanced use cases:

```python
from lighter.system import System

class CustomSystem(System):
    def _log_stats(self, loss, metrics, batch_idx):
        super()._log_stats(loss, metrics, batch_idx)
        # Add custom logging
        if self.mode == Mode.TRAIN:
            self.log("custom/my_metric", my_value)

    def on_train_epoch_end(self):
        # Custom behavior at epoch end
        pass
```

Use in config:

```yaml
system:
  _target_: project.CustomSystem
  model: ...
```

## Summary

System provides:

1. **Unified execution**: Same `_step()` for all modes
2. **Automatic pruning**: Unused components removed by stage
3. **Flexible loss**: Scalar or dict-based
4. **Smart injection**: Epoch/step passed to model automatically
5. **Inferer support**: Custom inference logic
6. **Comprehensive logging**: Loss, metrics, optimizer stats
7. **Extensibility**: Subclass for custom behavior

Understanding System helps you debug issues, optimize performance, and implement advanced training strategies.

## Next Steps

- [Adapters](../how-to/adapters.md) - Data transformation
- [Architecture Overview](overview.md) - High-level design
- [API Reference](../reference/) - Complete documentation
