---
title: CLI Reference
---

# CLI Reference

```bash
lighter {fit,validate,test,predict} INPUTS... [OPTIONS]
python -m lighter {fit,validate,test,predict} INPUTS... [OPTIONS]
lighter --version
```

`INPUTS` are config files and overrides, separated by spaces and applied left to right.

## Options

| Option | Stages | Meaning |
|--------|--------|---------|
| `--ckpt_path PATH` | all | Checkpoint to start from: a file, or `last` (see below) |
| `--weights_only` / `--no-weights_only` | all | Passed to Lightning's checkpoint loading |
| `--print_config` (`--print-config`) | all | Print the composed config as YAML and exit without building anything |
| `--verbose` / `--no-verbose` | validate, test | Print the results |
| `--return_predictions` / `--no-return_predictions` | predict | Keep the predictions in memory and return them |

`--ckpt_path last` resumes only from a `last*.ckpt` in this run's checkpoint folder. Each run logs to a new
`lightning_logs/version_N/` folder, so pass the checkpoint file, or give a `ModelCheckpoint` a fixed `dirpath`
and `save_last: true`.

## Overrides

| Override | Effect |
|----------|--------|
| `key::sub=value` | Set a value, written in YAML (`10`, `0.001`, `true`, `null`, `[1, 2]`, `{a: 1}`); dicts merge and lists are extended |
| `=key::sub=value` | Replace the value instead of merging into it |
| `~key::sub` | Delete a key |

```bash
lighter fit config.yaml model::optimizer::lr=0.01 trainer::max_epochs=100
lighter fit base.yaml experiment.yaml trainer::devices=2
lighter test config.yaml --ckpt_path lightning_logs/version_0/checkpoints/epoch=9-step=1000.ckpt
```

Quote overrides that hold `$`, `@`, spaces or brackets: `'model::lr=$@base_lr * 2'`.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | An error raised in your own code while building the run (your frames are shown) |
| 2 | A config or build error (the message starts with `file:line (key)` when the location is known) |

Errors raised inside the Lightning stage (e.g. in `training_step`) propagate unchanged.
`LIGHTER_DEBUG=1` shows full tracebacks.
