# Using LightningModule

Use ordinary Lightning modules with Lighter's configuration and records. The module continues to own its steps, logging, optimization and native hooks. This is the appropriate path for existing code and for behavior outside the managed LighterModule profile.

## Migration from pure Lightning

The project-discovery route needs `__lighter__.py` and `__init__.py` in the directory where you run the CLI. Normally installed packages can keep their usual module paths.

Move constructor arguments from your script into a recipe. This is a composition excerpt, assuming the named Python classes exist:

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 3
model:
  _target_: project.task.MyLightningModule
  learning_rate: 0.01
data:
  _target_: project.data.MyDataModule
```

Lighter does not rewrite constructors or translate a complete Python training script. Keep any necessary preparation and lifecycle behavior in your native module, data module or callbacks.

## Basic example

A native module for the diagnostic's dictionary batches can own SGD directly:

```python
import torch
import pytorch_lightning as pl


class RegressionTask(pl.LightningModule):
    def __init__(self, network, learning_rate=0.01):
        super().__init__()
        self.network = network
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        prediction = self(batch["x"]).squeeze(-1)
        loss = torch.nn.functional.mse_loss(prediction, batch["target"])
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.2)
```

This is a training-only module, not the full diagnostic replacement: add validation, test and prediction behavior before using those stages or a validation monitor. The existing [Compare and Continue native implementation](../examples/index.md) provides the complete research comparison.

## Hooks and custom optimization

Use Lightning's public hooks according to their lifecycle:

| Need | Native interface |
|---|---|
| Construct optimizer/scheduler | `configure_optimizers()` |
| Strategy-aware model construction | `configure_model()`, with a compatible strategy and module |
| Stage-dependent data setup | `LightningDataModule.prepare_data()` and `setup(stage)` |
| Observe restored optimizer settings | A runtime hook with access to `trainer.optimizers`, such as `on_train_start` |
| Manual or multiple-optimizer algorithm | `automatic_optimization = False`, `self.optimizers()`, `self.manual_backward()` |

Manual optimization owns zeroing, backward, clipping, optimizer/scheduler steps and any partial accumulation window. Configuration does not supply that algorithm. Use native optimizer wrappers so strategy/precision behavior participates correctly.

Changing an inherited LighterModule constructor or optimizer hook changes its ownership path; read [managed boundaries](lighter-module.md#construction-and-optimizer-ownership) before doing so. An arbitrary native constructor is not automatically made compatible with sharded construction.

## Measurements are part of the scientific task

Native modules call `self.log` or `self.log_dict` themselves. Configure callbacks to monitor the actual emitted name. For a batch mean, identify its denominator; an epoch mean of unequal batch means can be wrong without the intended weighting. Use stateful metrics or explicit population sums/counts for your task.

Keep [validation populations](custom-code.md#hold-out-validation-data), masking, multi-loader states and distributed reduction explicit. Native availability of a hook is not proof that a particular scientific implementation uses it correctly.

## Checkpoints and branches

A `fit` call with `ckpt_path` restores optimizer and loop state as well as model state. A changed constructor LR is therefore not automatically a new optimizer policy. To start a fresh optimization experiment, construct a fresh model/optimizer, load compatible model weights through ordinary Python, and fit without a resume checkpoint. See [checkpoint workflows](training.md#checkpointing).

## Public API and extension contracts

Public entrypoints include:

```python
from lighter import LighterDataModule, LighterModule, Runner
from lighter.callbacks import CsvWriter, FileWriter, Freezer
```

Runner accepts a stage name, a list of configuration inputs and native stage keyword arguments. For example, from the diagnostic project, this is the programmatic equivalent of a static configuration's prediction command after fitting:

```python
runner = Runner()
result = runner.run(
    "predict",
    ["config.yaml", "output_dir=outputs/first-experiment"],
    ckpt_path="outputs/first-experiment/checkpoints/last.ckpt",
    return_predictions=False,
)
print(runner.last_run_path)
```

Runner preserves native results and exceptions; its last record path can be absent when setup fails before recording. Calling this runs prediction, rather than merely describing a recipe.

Generated references cover [Runner](../reference/engine/runner.md), [LighterModule](../reference/model.md), [LighterDataModule](../reference/data.md) and [callbacks](../reference/callbacks/index.md). Prefer documented public methods and native Lightning hooks for extensions. Underscored helpers and generated internal entries are implementation details, not stable extension contracts.

For advanced strategies, tuning, precision and distributed execution, consult the [native Lightning documentation](https://lightning.ai/docs/pytorch/stable/) alongside Lighter's [tested-profile limits](compatibility.md#exact-validation-profiles). The native route remains available without claiming every configuration has been exercised.
