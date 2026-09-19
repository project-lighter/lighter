# Research practices

Use the smallest complete experiment that answers your question. Lighter helps make configuration and runtime observations visible; the scientific contract remains yours.

## Define the comparison

Name the decision before running: which setting changes, what stays fixed, what validation criterion selects the model and when final evaluation occurs. Retain every declared condition. A small experiment can be useful without proving a general ranking.

[Compare and Continue](../examples/index.md) shows this process with explicit populations, training budgets, checkpoint selection and matched native controls.

## Keep population and measurement semantics explicit

- Preserve sample IDs, split membership and preprocessing policy. Fit preprocessing using training data only.
- Define what the loss averages: examples, valid elements, sequences or another unit.
- Aggregate unequal batches with the intended denominator. Keep separate populations' metric state independent.
- Declare how partial batches, masks, empty populations and distributed sampling affect counts.
- Keep final-test observations out of tuning and checkpoint selection.

A seed is useful but does not freeze software, hardware or data. [Module measurements](lighter-module.md#automatic-logging) and [custom data](custom-code.md#hold-out-validation-data) explain Lighter-specific boundaries.

## Check behavior before scaling

Inspect the composed source first. Then use a small actual run to inspect shapes, IDs, losses and update counts. For a new algorithm, a tiny independently understood update is stronger evidence than a plausible training curve.

`fast_dev_run` executes training/validation batches; it is not a static check. Confirm that callbacks monitor emitted metric names. Manual optimization owns the update algorithm, including clipping and the last partial accumulation window.

Schedulers must use consistent units for duration, milestones and stepping. Precision, warmup, EMA and multiple optimizers are scientific choices implemented through native APIs; they are not universal recipes supplied by this guide.

## Save what explains the result

Retain the selected and last checkpoints separately, the population identities and predictions, source/code versions, dependency/environment identity and effective optimizer state. [Local records](experiment-records.md) preserve useful requested/observed context but do not prove an update or artifact's scientific correctness.

Use deliberate output directories. Lighter does not invent a timestamped experiment layout. For large exports, use [native streaming](predictions.md#stream-or-retain-native-outputs) and verify final IDs/counts.

## Separate continuation from a new experiment

A full checkpoint resume restores LR, momentum and progress. Loading only model weights into a fresh optimizer is a different experiment. Earlier-checkpoint fallback loses progress; do not label it exact recovery. Preserve the original exception and state identity when diagnosing failures.

See [checkpointing](training.md#checkpointing) and [freezing](freezing.md) before changing ownership or parameter groups.

## Keep maintenance proportionate

Start with a task, data policy and recipe, then factor repeated code when needed. Use supported [configuration composition](configuration.md) rather than hidden mutation. Keep one authoritative installation path and exact dependency evidence; use the [current-pair guide](compatibility.md) instead of unrelated example version pins.

Tests should check meaningful scientific or lifecycle behavior, not merely that a configuration object exists. The repository's [diagnostic](../quickstart.md#verify-the-same-workflow) checks actual checkpoint values, prediction identities and continuation. New tasks need their own checks; deployment, accelerator and usability claims require separate evidence.
