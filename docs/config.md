---
title: Config cheat sheet
---

# Config cheat sheet

A Lighter config is YAML that [Sparkwheel](https://project-lighter.github.io/sparkwheel/) turns into Python
objects. It needs `trainer:` and `model:`, usually `data:`, and may hold any other keys you want to refer to.

## Build objects

| Write | Means |
|-------|-------|
| `_target_: a.b.C` | Call `a.b.C(...)` with the other keys as keyword arguments |
| `_args_: [...]` | Positional arguments for that call |
| `_mode_: callable` | Do not call it: build `functools.partial(a.b.C, ...)`, to be called later |
| `_disabled_: true` | Leave this object out (in a list, the item is removed) |
| `_imports_: {name: module}` | Modules that `$` expressions can use (top level only) |

```yaml
network:
  _target_: torch.nn.Sequential
  _args_:
    - _target_: torch.nn.Linear
      in_features: 8
      out_features: 32
    - _target_: torch.nn.ReLU
    - _target_: torch.nn.Linear
      in_features: 32
      out_features: 1

optimizer:
  _target_: torch.optim.Adam  # LighterModule calls this partial with the model's parameters
  _mode_: callable
  lr: 0.001

callbacks:
  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
  - _target_: pytorch_lightning.callbacks.EarlyStopping
    _disabled_: true  # switched off, so the list holds one callback
    monitor: val/loss/epoch
```

## Refer to other values

| Write | Means |
|-------|-------|
| `"@key::sub"` | The value built at `key::sub`. Every `@` to it gets the same object |
| `"%key::sub"` | A copy of the definition at `key::sub`, built again as a new object |
| `"$expression"` | A Python expression. Inside it, read other values with `@key` |
| `key::sub` | A path into the config: `::` separates keys |

```yaml
_imports_:
  math: math

num_classes: 10
batch_size: 32
steps_per_epoch: "$math.ceil(50000 / @batch_size)"

train_metrics:
  _target_: torchmetrics.Accuracy
  task: multiclass
  num_classes: "@num_classes"
val_metrics: "%train_metrics"  # its own Accuracy: "@train_metrics" would share one object and mix the counts
```

## Combine files and overrides

`lighter fit a.yaml b.yaml key=value` applies the inputs from left to right. The same three operations work in a
file stacked after another one and as an override on the command line:

| In a file | On the command line | Effect |
|-----------|---------------------|--------|
| `key: value` | `key::sub=value` | Set. Dicts are merged key by key and lists are extended |
| `=key: value` | `=key::sub=value` | Replace the value instead of merging into it |
| `~key: null` | `~key::sub` | Delete the key (it must exist) |

`--print_config` prints the composed config and stops, which is the quickest way to check what an override did.

```bash
lighter fit config.yaml experiment.yaml model::lr=0.001 trainer::max_epochs=20
lighter fit config.yaml '=trainer::callbacks=[]' --print_config
lighter fit config.yaml ~data::val_dataloader
```

## Values and the shell

- An override's value is YAML: `10`, `0.001`, `true`, `null`, `[1, 2]`, `{a: 1}`; anything else is a string.
- Write floats with a dot: in YAML 1.1, which Lighter reads, `1e-3` is the string `"1e-3"`. Write `0.001` or
  `1.0e-3`, in files and in overrides.
- A list override extends the list: `'trainer::callbacks=[...]'` adds callbacks. Put `=` in front to replace it.
- An override may add a key that the config does not have, so a misspelled top-level key is accepted
  silently. A misspelled argument under a `_target_` fails when the object is built.
- Quote an override in single quotes when it holds `$` (the shell would expand it), spaces, brackets (zsh
  treats `[...]` as a file pattern) or braces. `@` and `::` need no quotes. These rules are for bash and zsh.

```bash
lighter fit config.yaml 'trainer::devices=[0, 1]' 'model::lr=$@base_lr * 10'
```
