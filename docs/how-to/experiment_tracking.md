# Experiment Tracking

Configure loggers to track experiments.

## TensorBoard (Built-in)

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

## Related Guides
- [Run Guide](run.md) - Running experiments
- [Troubleshooting](troubleshooting.md) - Common issues
