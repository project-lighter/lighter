# Using LighterModule

LighterModule supplies a default forward pass, managed optimizer/scheduler setup and automatic measurements. You implement the scientific steps and update configured metrics. Start with its inherited constructor; ordinary authoring needs no factory or setup hook.

Use a [native LightningModule](lightning-module.md) when your module owns a different lifecycle or optimization algorithm.

## Required implementations

The default `forward(*args, **kwargs)` calls `self.network`. Implement `training_step` for training and the other stages your task needs. This excerpt is from the [download-free diagnostic](../quickstart.md):

```python
from lighter import LighterModule


class RegressionTask(LighterModule):
    def training_step(self, batch, batch_idx):
        prediction = self(batch["x"]).squeeze(-1)
        return self.criterion(prediction, batch["target"])
```

The complete project also implements validation, test and prediction. Omitting a validation/test method leaves that stage absent according to native detection; returning `None` from an implemented evaluation method still allows explicit logging and configured metric collection. Implement a prediction step that preserves your sample IDs.

## Construction and optimizer ownership

In the ordinary managed path, the inherited constructor builds the network, criterion and metrics eagerly. Lighter retains the optimizer/scheduler recipes and resolves them at Lightning's native `configure_optimizers()` setup point:

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
    momentum: 0.2
```

This is the diagnostic's model section with an LR change. Each native optimizer setup receives fresh parameter iterators, optimizer and scheduler objects against the actual network. Parameter groups, subsets, ordering and options remain those selected by the recipe. Native restoration then loads saved optimizer/scheduler state. The network remains the same object across one module instance's stages; separate Runner runs ordinarily construct separate modules.

The managed path requires a statically named class inheriting both `LighterModule.__init__` and `configure_optimizers`, with ordinary keyword component definitions. Local `%` copies are supported. Custom constructors/hooks, positional module construction and computed class targets retain eager/native ownership.

A prebuilt optimizer, scheduler, parameter or iterator leaf anywhere in retained source conservatively preserves eager/native ownership, including aliases and unused helper leaves. This source check never evaluates inactive definitions or consumes opaque objects. A prebuilt network module alone can still use managed optimizer setup.

### Requested values versus live objects

An eager callback cannot require an optimizer reserved for later native setup. Share the requested scalar instead:

```yaml
learning_rate: 0.01
model:
  optimizer:
    lr: "@learning_rate"
```

This is an overlay for an existing recipe. An eager callback may also use `@learning_rate`. Referencing `@model::optimizer::lr` enters the blocked optimizer subtree and fails with its source path.

For ordinary `Trainer.fit`, observe the **effective restored** LR through `trainer.optimizers` in `on_train_start`, after native setup and checkpoint restoration. A custom lifecycle owns when those objects are available. [Compare and Continue](../examples/index.md) and [records](experiment-records.md#requested-settings-and-restored-results) demonstrate the distinction.

### Custom construction and strategies

A managed optimizer can see parameters replaced inside the existing network during `configure_model`, because its parameter expression resolves afterward. Replacing the entire managed network, or removing/replacing a separately constructed shared child/parameter, raises an alias-conflict diagnostic. Use custom native ownership for those patterns. Lighter does not rewrite arbitrary aliases or constructors, and eager network construction does not certify sharded construction.

Migration: code that read `model.optimizer` before native setup must move that dependency to a runtime hook or keep custom ownership. Directly constructed modules and supplied optimizer objects retain their native lifetime.

## Automatic logging

The module logs returned loss observations, configured metrics and optimizer settings. `trainer.logger: false` disables external logging, not callback measurements. Monitor actual keys in `trainer.callback_metrics`; logger-disabled values may not appear in native `validate()`/`test()` return dictionaries.

### Loss logging

Under automatic optimization, return a single-element Tensor or a dictionary whose `loss` is that Tensor. Return `None` to skip a step. Use `loss_terms` for additional named scalar observations:

```python
return {"loss": total_loss, "loss_terms": {"data": data_loss, "penalty": penalty}}
```

This step excerpt assumes those tensors were computed by your algorithm. Automatic names include:

| Observation | Names |
|---|---|
| Training scalar loss | `train/loss/step`, `train/loss/epoch` |
| Validation scalar loss | `val/loss/step`, `val/loss/epoch` |
| Named training term | `train/loss/data/step`, `train/loss/data/epoch` |

The training observation is captured before Lightning divides closure loss for gradient accumulation. This preserves the returned scientific loss without changing gradients or optimization weighting. A nested dictionary under the automatic training `loss` is invalid; put named observations under `loss_terms`.

Epoch aggregation follows native logging behavior. A batch mean over valid tokens, pixels or masked elements is not necessarily an equally weighted mean over examples. For task-specific denominators, use explicit native metrics or sums/counts; automatic logging does not infer the scientific population.

### Metric logging

Configure a `torchmetrics.Metric` or `MetricCollection`, not a bare list. Call the metric in your step with its required inputs. For a regression step using a configured MeanSquaredError:

```python
self.val_metrics(prediction, batch["target"])
return self.criterion(prediction, batch["target"])
```

The metric is automatically logged under `val/metrics/MeanSquaredError/step` and `val/metrics/MeanSquaredError/epoch`. A MetricCollection uses each collection key, preserving its prefix/postfix. Metric epoch values come from the metric's state/computation; they are not universally averages of batch scores.

Use `%` to construct independent train/validation metric state. A copied definition can still contain `@` references to intentionally shared dependencies.

### Separate evaluation populations

With multiple validation/test loaders, accept `dataloader_idx` and update the metric normally:

```python
def validation_step(self, batch, batch_idx, dataloader_idx=0):
    prediction = self(batch["x"]).squeeze(-1)
    self.val_metrics(prediction, batch["target"])
    return self.criterion(prediction, batch["target"])
```

Inside the step, `self.val_metrics` refers to that loader's independent state; test behaves similarly. Lightning adds `/dataloader_idx_0`, `/dataloader_idx_1`, and so on. Reordering loaders changes their population-index mapping, not their pooling policy. Successive evaluations reset through native logging, including sanity validation.

Outside a step, these attributes refer to loader 0. Read per-loader epoch values through `trainer.callback_metrics`. Automatic cloning for multiple loaders currently requires an empty metric `state_dict()`; checkpoint-bearing/persistent metric state is rejected rather than discarded. Use explicit per-loader native metric attributes for such cases or for manual `self.log` ownership. Do not mix automatic managed attributes with custom manual logging that caches their identity before later loader states exist.

Earlier versions could pool separate populations and rescale logged training loss during accumulation. Those fixes do not rewrite historical measurements.

## Schedulers and manual ownership

The constructor accepts a native scheduler or a Lightning scheduler dictionary. This is a scheduler-section excerpt for an existing managed optimizer:

```yaml
model:
  scheduler:
    scheduler:
      _target_: torch.optim.lr_scheduler.StepLR
      optimizer: "@model::optimizer"
      step_size: 10
    interval: epoch
    frequency: 1
```

Native Lightning owns stepping for automatic optimization. A plateau scheduler additionally needs its actual monitor. Choose epoch or optimizer-step units deliberately; a warmup's length and transition must use the same unit.

Manual optimization remains manual: own backward, zeroing, clipping, updates and scheduler stepping. Return the scientific loss you want observed, not an internal scaled backward value. Native/custom optimizer hooks remain available without an automatic-ownership promise.

LighterModule's inherited batch-end hooks perform automatic logging. If you override those hooks, preserve the inherited behavior you still need. Prefer native public hooks over underscored implementation helpers. See the [public API and extension contracts](lightning-module.md#public-api-and-extension-contracts).
