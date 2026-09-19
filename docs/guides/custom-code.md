# Custom code

Keep datasets, model behavior and scientific decisions in Python. YAML describes the objects and settings that connect them.

## The project folder pattern

Run Lighter from the directory containing an empty `__lighter__.py` marker and `__init__.py`:

```text
my_experiment/
├── __lighter__.py
├── __init__.py
├── task.py
├── data.py
└── config.yaml
```

Runner imports this directory as `project`. A class in `task.py` is therefore `project.task.MyTask`. Subpackages need their own `__init__.py`. The marker is not a setup script; package initialization occurs after the configured seed.

Normally installed Python packages can use their ordinary import paths instead. Lighter does not require a marker for an already importable module.

## Start from working code

The [download-free project](../quickstart.md) supplies a complete model, data and recipe. Its `RegressionTask` inherits LighterModule's constructor and optimizer hook; only its scientific steps are written in the subclass.

To adapt it, change the dataset and step together. Its batch contract is a dictionary containing `x`, `target` and stable `id` values. A different batch shape is your Python contract, not a YAML convention.

For example, this is a step excerpt for that batch shape:

```python
def training_step(self, batch, batch_idx):
    prediction = self(batch["x"]).squeeze(-1)
    return self.criterion(prediction, batch["target"])
```

Use a [native LightningModule](lightning-module.md) when your module owns custom optimization or setup. Use [LighterModule](lighter-module.md) when its inherited constructor and automatic measurements fit your task. Neither path requires LightningCLI.

## Hold out validation data

Define populations before tuning. Fit preprocessing on training data only; use validation for model/checkpoint selection; reserve test data for the final declared comparison. Preserve stable sample IDs through batching and export.

The diagnostic uses disjoint generated sample indices. [Compare and Continue](../examples/index.md) uses fixed, disjoint CIFAR-10 train/validation populations and a separately selected official-test population, recording their exact identities. Its data preparation also declares tensor shape, dtype and normalization.

For your own data, make split membership, transforms, masking and aggregation explicit. A train/eval mode change controls behavior such as dropout; it does not establish a held-out population. Variable-length samples need a deliberate choice between per-example and per-valid-element weighting.

## Data preparation and loaders

LighterDataModule wraps **already constructed** loaders and returns them through native hooks. Its constructor accepts `train_dataloader`, `val_dataloader`, `test_dataloader` and `predict_dataloader`. Omitted loaders are absent stages according to native Lightning detection.

A data-section excerpt from the diagnostic is:

```yaml
data:
  _target_: lighter.LighterDataModule
  train_dataloader:
    _target_: torch.utils.data.DataLoader
    dataset:
      _target_: project.task.RegressionSamples
      split: train
    batch_size: 8
    shuffle: true
```

Configured dataset and loader construction is eager. If downloading/shared preparation belongs in `prepare_data()`, or loader construction depends on the stage in `setup(stage)`, use a native LightningDataModule. Keep worker seeding, distributed sampling, IDs and output ownership consistent with the selected execution strategy.

## Organize only what your experiment needs

Start with one scientific task, a data definition and a recipe. Add small overlays for experiment choices. Separate reusable Python components once there is an actual second consumer; a large directory hierarchy is not a prerequisite.

Use configuration `@` references for intentional object sharing and `%` copies for independent construction. These meanings do not replace Python ownership checks: if two heads should share one encoder, they must receive that same object, and optimizer membership must reflect your intended parameters.

## Troubleshooting

| Symptom | Check |
|---|---|
| `ModuleNotFoundError: project` | Current directory, both marker files and the recipe's module path |
| Class cannot be imported | Class spelling, containing Python module and package initialization |
| A loader performs work too early | Move stage-dependent work into native data hooks |
| Optimizer reference blocked during construction | Use a shared requested scalar or a native runtime hook; see [ownership](lighter-module.md#construction-and-optimizer-ownership) |
| Predictions no longer match samples | Preserve IDs and check lengths/order at the data-to-output boundary |

Read [configuration](configuration.md) for composition and [research practices](best-practices.md) for the checks that accompany a new scientific task.
