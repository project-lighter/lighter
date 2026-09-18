---
title: Quick Start
---

# Quick Start

Start with a complete experiment that needs no dataset download or external logger. The [tabular regression example](https://github.com/project-lighter/lighter/tree/main/projects/tabular_regression) fits a small model, evaluates a checkpoint, saves identifiable predictions and resumes training on CPU.

## Install and run

From a checkout with matching Lighter and Sparkwheel dependencies:

```bash
pip install -e .
cd projects/tabular_regression
lighter fit config.yaml model::optimizer::lr=0.05
lighter test config.yaml --ckpt-path outputs/tabular_regression/checkpoints/last.ckpt --no-verbose
lighter predict config.yaml --ckpt-path outputs/tabular_regression/checkpoints/last.ckpt --no-return-predictions
lighter fit config.yaml --ckpt-path outputs/tabular_regression/checkpoints/last.ckpt trainer::max_epochs=5
```

`python -m lighter` invokes the same CLI. Each command runs the named stage. In a new process, name a concrete checkpoint file instead of assuming that `best` or `last` can discover a previous run.

The example explicitly enables `ModelCheckpoint(save_last=True)`, supplies CSV output paths and columns, and uses `logger: false`. Callback metrics and checkpoint monitoring remain active. Output locations come from the recipe; no timestamped directory convention is implied.

To run those four CLI stages and verify the results automatically:

```bash
python workflow.py --output-dir outputs/verified
```

This checks checkpoint progress from 9 to 15 optimizer steps and exact coverage of all seven prediction IDs, including leading zeros and Unicode. Logs and a machine-readable `workflow.json` stay in the selected output directory. [Local run records](guides/experiment-records.md) provide inspection across separate invocations.

## Familiar Python, configurable components

A project directory contains an empty `__lighter__.py` marker and `__init__.py`. Lighter imports that directory as `project`, so `task.py` is available as `project.task`.

```python
from lighter import LighterModule


class RegressionTask(LighterModule):
    def training_step(self, batch, batch_idx):
        prediction = self(batch["x"]).squeeze(-1)
        return self.criterion(prediction, batch["target"])
```

The [complete example](https://github.com/project-lighter/lighter/tree/main/projects/tabular_regression) adds validation, test and prediction steps plus a tiny dataset. Its optimizer recipe stays ordinary:

```yaml
model:
  _target_: project.task.RegressionTask
  network:
    _target_: torch.nn.Linear
    in_features: 2
    out_features: 1
  criterion:
    _target_: torch.nn.MSELoss
  optimizer:
    _target_: torch.optim.SGD
    params: "$@model::network.parameters()"
    lr: 0.05
```

The inherited Lighter constructor builds the network normally; managed optimizer construction happens at Lightning's native setup point. No user factory or extra hook is needed. Native Lightning modules and custom optimizer hooks retain their own ownership. See the [module guide](guides/lighter-module.md#construction-and-optimizer-ownership) for exact boundaries.

Override nested choices with `::`, and pass multiple files as separate arguments:

```bash
lighter fit config.yaml trainer::max_epochs=10 seed=42
lighter fit base.yaml experiment.yaml model::optimizer::lr=0.001
```

Keep algorithm-specific logic in Python. Use [native Lightning modules](guides/lightning-module.md) for custom ownership and multiple optimizers, [configuration composition](guides/configuration.md) for reusable recipes, and [prediction writers](guides/predictions.md) for artifacts.
