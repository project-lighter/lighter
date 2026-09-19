# Configuration Guide

A recipe composes Python objects and values. Lighter uses Sparkwheel for this composition, then passes the resulting objects to native Lightning. Scientific behavior stays in your module and data code.

For the pinned installation snapshot, read the [matching Sparkwheel reference](https://github.com/project-lighter/sparkwheel/tree/b73e786e8716d11a77206fb3482d21a621a4ba81/docs/user-guide). Links to the [upstream Sparkwheel site](https://project-lighter.github.io/sparkwheel/) may describe a different released version.

Use the existing [quick-start project](../quickstart.md) for a complete runnable recipe. The YAML below illustrates individual sections and overlays.

## Config structure

| Key | Purpose |
|---|---|
| `model` | A LightningModule, including a LighterModule subclass |
| `trainer` | A native `pytorch_lightning.Trainer` |
| `data` | Optional native LightningDataModule or LighterDataModule |
| `args` | Native stage arguments, such as `args::test::ckpt_path` |
| `seed` | Literal integer applied before project imports and construction |
| `run` | Optional local record settings; `false` disables records |

You can add ordinary top-level settings, such as `learning_rate` or `output_dir`, and reference them elsewhere. `model` owns the module: a stage argument cannot replace it. A native loader may instead be supplied through an appropriate `args::<stage>` entry.

## The essential symbols

| Syntax | Meaning | Example |
|---|---|---|
| `_target_` | Import and call a Python target | `_target_: torch.nn.Linear` |
| `_args_` | Positional arguments to that target | `_args_: [4, 1]` |
| `@` | Resolve and reuse a value/object | `lr: "@learning_rate"` |
| `%` | Copy a source definition for separate resolution | `val_metrics: "%model::train_metrics"` |
| `$` | Evaluate a Python expression | `params: "$@model::network.parameters()"` |
| `::` | Address a configuration path | `model::optimizer::lr=0.05` |

### Create objects

```yaml
network:
  _target_: torch.nn.Linear
  in_features: 2
  out_features: 1
```

This describes `torch.nn.Linear(in_features=2, out_features=1)`. The target can be your own importable class. Project-local imports use `project.*` after [project discovery](custom-code.md#the-project-folder-pattern).

### Share a value or copy a definition

`@` shares the resolved object within its resolution context. Use it when two consumers should see the same network, optimizer or scalar. `%` copies the source definition; if that definition constructs an object, resolving the copy normally constructs a separate object. A copied definition can still contain `@` references to shared dependencies.

For independent metric state, use a definition copy:

```yaml
model:
  train_metrics:
    _target_: torchmetrics.MeanMetric
  val_metrics: "%model::train_metrics"
```

This is a model-section excerpt. Your step must update each metric with values appropriate to its definition. A MeanMetric takes observations; a classification Accuracy metric takes predictions and targets. Neither call is supplied by configuration.

### Expressions and Python attributes

```yaml
learning_rate: 0.01
model:
  optimizer:
    _target_: torch.optim.SGD
    params: "$@model::network.parameters()"
    lr: "@learning_rate"
```

This overlay assumes an existing managed model/network recipe. `::` navigates configuration; `.` accesses the resulting Python object. Write `network.parameters()`, not `network::parameters()`.

Quote expressions and operator-prefixed strings in YAML. An expression is Python code; use trusted recipes and targets. Loading/composing source differs from resolving expressions and constructing objects. A reference does not promise that an optimizer exists before Lightning setup; see [construction ownership](lighter-module.md#construction-and-optimizer-ownership).

## CLI overrides

From the diagnostic project, inspect a changed definition:

```bash
python -m lighter inspect config.yaml model::optimizer::lr=0.05 trainer::max_epochs=5 --json
```

CLI override values follow Sparkwheel's YAML parsing. Quote the entire argument when shell metacharacters or spaces occur. The inspected source still contains expressions; it is not a dump of live runtime objects.

## Merging configs

Pass files as separate arguments; later inputs compose over earlier ones. This pattern assumes files you created:

```bash
python -m lighter inspect base.yaml experiment.yaml --json
```

Mappings merge recursively, ordinary lists extend, and scalar values replace earlier values. For example, an overlay containing only `trainer.max_epochs` retains other Trainer settings. To replace or delete rather than merge, use the explicit operators:

```yaml
trainer:
  =callbacks: []       # Replace the callback list
  ~limit_train_batches: null  # Delete this key
```

Indexed deletion uses `~callbacks: [1, 3]` to remove those list entries. Review the composed source before execution, especially when combining callback lists. Full operator semantics are in the matching `sparkwheel/docs/user-guide/operators.md` ([upstream guide](https://project-lighter.github.io/sparkwheel/user-guide/operators/)).

## Functions, disabled components and imports

`_mode_: callable` creates a partial callable instead of calling the target immediately. Use it for an API that expects a callable, such as a DataLoader's `collate_fn`; ordinary model/optimizer authoring does not require a factory.

`_disabled_: true` removes an inline component from its parent container; directly resolving it or referring to it with `@` yields `None`. `_imports_` supplies names to expressions. These are composition features, not a replacement for native lifecycle hooks. See the matching `sparkwheel/docs/user-guide/instantiation.md` ([upstream reference](https://project-lighter.github.io/sparkwheel/user-guide/instantiation/)) for detailed modes and boundaries.

## Complete example

Use `projects/tabular_regression/config.yaml` with its accompanying `task.py`; the [quick start](../quickstart.md) shows its complete execution. A bare LighterModule is not a completed scientific task: you must implement its training step. Configure a validation loader and validation step before monitoring a validation metric.

## Importing Lighter in a host application

`import lighter` preserves application logging handlers, warning/exception hooks and multiprocessing serialization. Configure host logging explicitly. Programmatic use is through [Runner and native APIs](lightning-module.md#public-api-and-extension-contracts).

The `__lighter__.py` marker is not executed. Runner imports the enclosing package's `__init__.py` after seeding. Explicit path imports activate Lighter's dynamic-module finder and multiprocessing serialization support so project classes can reach spawned workers. Use normally installed/importable packages when a host needs to retain its own serializer. Importing an already-loaded module name from another directory fails rather than silently selecting the wrong project.
