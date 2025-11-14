# Experiment Tracking

Lighter provides comprehensive experiment tracking through PyTorch Lightning loggers and Writer callbacks. This guide covers what's logged automatically, how to configure loggers, and how to add custom logging.

## What Lighter Logs Automatically

Lighter's System class automatically logs the following metrics without any configuration:

### 1. Loss Values

**Logged in**: train and val modes (test mode doesn't compute loss by default)

```
{mode}/loss/step    # Per-batch loss
{mode}/loss/epoch   # Epoch-averaged loss
```

For dict-based losses (multi-task learning), all sublosses are logged:

```
{mode}/loss/total/step
{mode}/loss/total/epoch
{mode}/loss/classification/step
{mode}/loss/classification/epoch
{mode}/loss/segmentation/step
{mode}/loss/segmentation/epoch
```

**Code reference**: `src/lighter/system.py:194-202`

### 2. Metrics

**Logged in**: train, val, and test modes (predict mode doesn't compute metrics)

```
{mode}/metrics/{metric_name}/step
{mode}/metrics/{metric_name}/epoch
```

All metrics defined in your config are automatically logged with both step-level and epoch-level aggregation.

**Code reference**: `src/lighter/system.py:205-208`

### 3. Optimizer Statistics

**Logged in**: train mode only, **once per epoch** (at the beginning)

```
train/lr         # Learning rate
train/momentum   # If using SGD with momentum
train/beta1      # If using Adam/AdamW
train/beta2      # If using Adam/AdamW
```

This automatic logging helps you track learning rate schedules and optimizer behavior without any additional configuration.

**Code reference**: `src/lighter/system.py:210-213`

### 4. Hyperparameters

The Runner automatically logs all configuration parameters (the entire YAML config) to the logger at the start of training. This ensures full reproducibility.

**Code reference**: `src/lighter/engine/runner.py:143-146`

## Logger Configuration

Lighter uses PyTorch Lightning's logger system. You can configure any Lightning-compatible logger in your config.

### TensorBoard (Built-in)

```yaml
trainer:
  logger:
    _target_: pytorch_lightning.loggers.TensorBoardLogger
    save_dir: logs
    name: my_experiment
    version: null  # Auto-incrementing version
```

```bash
# View logs
tensorboard --logdir logs
```

## Weights & Biases

```yaml
trainer:
  logger:
    _target_: pytorch_lightning.loggers.WandbLogger
    project: my_project
    name: experiment_name
    save_dir: logs
```

## MLflow

```yaml
trainer:
  logger:
    _target_: pytorch_lightning.loggers.MLFlowLogger
    experiment_name: my_experiment
    tracking_uri: file:./mlruns
```

## CSV Logger

```yaml
trainer:
  logger:
    _target_: pytorch_lightning.loggers.CSVLogger
    save_dir: logs
    name: my_experiment
```

## Multiple Loggers

```yaml
trainer:
  logger:
    - _target_: pytorch_lightning.loggers.TensorBoardLogger
      save_dir: logs
    - _target_: pytorch_lightning.loggers.WandbLogger
      project: my_project
```

For advanced logging features, see [PyTorch Lightning Logger docs](https://lightning.ai/docs/pytorch/stable/extensions/logging.html).

## Writer Callbacks for Predictions

While loggers handle scalar metrics, Writer callbacks save predictions, inputs, and targets to disk. Writers are triggered automatically after each batch in val, test, and predict modes.

### Available Writers

Lighter provides several built-in writers in `lighter.callbacks.writer`:

- **FileWriter**: Save individual files (images, arrays)
- **TableWriter**: Save predictions in tabular format (CSV, Parquet)

For detailed usage, see the [Writers Guide](writers.md).

### When Writers Are Triggered

Writers are **batch-level callbacks** that run after each batch in val/test/predict modes:

```
For each batch in validation/test/predict:
  1. System._step() computes predictions
  2. Output dict is returned to callbacks
  3. Writer callbacks process the dict
  4. Files are written to disk
```

This allows you to save all predictions without accumulating them in memory.

### Writer Memory Management

Writers actively clear predictions from the output dictionary after processing to save CPU memory. This is especially important for large-scale inference.

**Code reference**: `src/lighter/callbacks/writer.py:141-143`

## Custom Logging Strategies

### Strategy 1: Logging Additional Scalars

To log custom values, extend the System class and override `_log_stats`:

```python title="my_project/custom_system.py"
from lighter.system import System

class CustomSystem(System):
    def _log_stats(self, loss, metrics, batch_idx):
        # Call parent to log standard metrics
        super()._log_stats(loss, metrics, batch_idx)

        # Log custom values
        if self.mode == "train":
            # Example: Log gradient norms
            grad_norm = self._compute_gradient_norm()
            self.log(f"{self.mode}/grad_norm", grad_norm)

            # Example: Log model statistics
            self.log(f"{self.mode}/model_mean_weight",
                    self.model.fc.weight.mean())

    def _compute_gradient_norm(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
```

Use in config:

```yaml
system:
  _target_: my_project.custom_system.CustomSystem
  model: ...
  # ... other components
```

### Strategy 2: Conditional Logging

Log different metrics based on mode or epoch:

```python title="my_project/conditional_system.py"
from lighter.system import System

class ConditionalSystem(System):
    def _log_stats(self, loss, metrics, batch_idx):
        super()._log_stats(loss, metrics, batch_idx)

        # Only log expensive metrics every N epochs
        if self.current_epoch % 10 == 0:
            if self.mode == "val":
                self.log(f"{self.mode}/expensive_metric",
                        self._compute_expensive_metric())

    def _compute_expensive_metric(self):
        # Expensive computation here
        pass
```

### Strategy 3: Logging Images/Media

For image logging, use the logger directly:

```python title="my_project/vision_system.py"
from lighter.system import System
import torch

class VisionSystem(System):
    def on_validation_epoch_end(self):
        # Log sample predictions as images
        if self.trainer.logger is not None:
            sample_input = self.validation_samples[:8]  # 8 images
            sample_pred = self.model(sample_input)

            # Log to TensorBoard
            if hasattr(self.trainer.logger.experiment, 'add_images'):
                self.trainer.logger.experiment.add_images(
                    'val/predictions',
                    sample_pred,
                    self.current_epoch
                )
```

## Integration: System, Logger, and Writers

Understanding how these components work together:

```
Training Loop (System)
  ↓
Automatic Logging (System._log_stats)
  ├─→ Logger receives scalar metrics
  └─→ TensorBoard/W&B/MLflow displays them
  ↓
Step Output (System._step returns dict)
  ↓
Callbacks (Writers)
  └─→ Save predictions/inputs/targets to files
```

### Example: Complete Tracking Setup

```yaml title="complete_tracking.yaml"
trainer:
  _target_: pytorch_lightning.Trainer

  # Multiple loggers for comprehensive tracking
  logger:
    - _target_: pytorch_lightning.loggers.TensorBoardLogger
      save_dir: logs
      name: my_experiment
      version: null

    - _target_: pytorch_lightning.loggers.WandbLogger
      project: my_project
      name: my_experiment
      save_dir: logs
      log_model: true

  # Logging frequency
  log_every_n_steps: 50

  # Callbacks for saving predictions
  callbacks:
    - _target_: pytorch_lightning.callbacks.LearningRateMonitor
      logging_interval: epoch

    - _target_: lighter.callbacks.writer.FileWriter
      write_dir: predictions
      predicates: ["val", "test"]

system:
  _target_: lighter.System
  # ... rest of config
```

## Advanced: Logger-Specific Features

### TensorBoard: Hyperparameter Tuning

TensorBoard can visualize hyperparameter search results:

```yaml
trainer:
  logger:
    _target_: pytorch_lightning.loggers.TensorBoardLogger
    save_dir: logs
    name: hparam_search
    default_hp_metric: true  # Enable HP tracking

# Vary hyperparameters across runs
system:
  optimizer:
    lr: 0.001  # Change this across experiments
```

View in TensorBoard:
```bash
tensorboard --logdir logs
# Navigate to HPARAMS tab
```

### W&B: Artifact Tracking

Track model checkpoints as W&B artifacts:

```yaml
trainer:
  logger:
    _target_: pytorch_lightning.loggers.WandbLogger
    project: my_project
    log_model: all  # 'all', True, False

  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      dirpath: checkpoints
      save_top_k: 3
```

### MLflow: Model Registry

Integrate with MLflow's model registry:

```yaml
trainer:
  logger:
    _target_: pytorch_lightning.loggers.MLFlowLogger
    experiment_name: my_experiment
    tracking_uri: file:./mlruns
    log_model: true  # Log as MLflow model
```

## Troubleshooting

### Issue: No logs appearing

**Solution**: Check that logger is not None:

```yaml
trainer:
  logger: ...  # Make sure this is configured
```

### Issue: Metrics not syncing across GPUs

**Solution**: Lighter automatically sets `sync_dist=True` for epoch-level metrics. For custom metrics, ensure you use `on_epoch=True`:

```python
self.log("custom_metric", value, on_epoch=True, sync_dist=True)
```

### Issue: Too much logging slowing down training

**Solution**: Reduce logging frequency:

```yaml
trainer:
  log_every_n_steps: 100  # Default is 50
```

### Issue: Writer memory usage too high

**Solution**: Writers automatically clear predictions. If still high, process predictions in smaller batches:

```yaml
system:
  dataloaders:
    predict:
      batch_size: 16  # Reduce batch size
```

## Best Practices

1. **Use multiple loggers**: Combine TensorBoard (local) with W&B/MLflow (team)
2. **Log hyperparameters**: Automatic in Lighter, but verify they appear in your logger
3. **Monitor optimizer stats**: Use `LearningRateMonitor` callback for detailed tracking
4. **Separate concerns**: Use Logger for scalars, Writers for predictions
5. **Version your experiments**: Use timestamps or version numbers in logger names
6. **Document runs**: Add notes/tags to experiments in W&B or MLflow

## Summary

Lighter provides comprehensive tracking out of the box:

- **Automatic**: Loss, metrics, optimizer stats, hyperparameters
- **Flexible**: Support for all PyTorch Lightning loggers
- **Scalable**: Batch-level Writers prevent memory issues
- **Extensible**: Easy to add custom logging via System subclassing

For most use cases, you just need to configure a logger—everything else is automatic!

## Related Guides
- [System Internals](../design/system.md) - Understanding automatic logging
- [Writers](writers.md) - Saving predictions to disk
- [Configuration Guide](configuration.md) - Logger configuration syntax
- [Run Guide](run.md) - Running experiments
