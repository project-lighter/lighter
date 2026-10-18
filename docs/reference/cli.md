---
title: CLI Reference
---

# CLI Reference

`lighter <stage> INPUTS... [OPTIONS]`, or `python -m lighter <stage> ...`. `INPUTS` are config files and
overrides, applied from left to right; the [config cheat sheet](../config.md#combine-files-and-overrides) shows
how they combine. This is the CLI's own help, which a test keeps in step with this page:

```text
--8<-- "tests/unit/cli_help.txt"
```

## `--ckpt_path last`

Each run logs to a new `lightning_logs/version_N/` folder, whose `checkpoints/` is empty when the run starts.
For `last` to find the previous run's checkpoint, give the `ModelCheckpoint` a fixed `dirpath` and
`save_last: true`, or pass the checkpoint file instead.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Your own code raised while the run was being built (only your frames are shown) |
| 2 | The config or the command line is wrong (the message starts with `file:line (key)` when the place is known) |

Errors raised during the stage itself, for example in `training_step`, are Lightning's and pass through
unchanged.
