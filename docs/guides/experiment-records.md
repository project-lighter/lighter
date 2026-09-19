# Inspecting and comparing experiments

Runner records each stage attempt locally, even with `trainer.logger: false`. Use native Lightning loggers for live dashboards and remote tracking. A local record is a portable summary of one attempt.

For the complete runnable inspect/list/show/diff sequence, follow the [quick start](../quickstart.md#inspect-the-attempts). Inspection composes source, including declared file includes, without evaluating configured expressions/imports/constructors. Record commands only read JSON; they do not replay recipes.

## Record location and identity

The default directory is `trainer.default_root_dir/lighter_runs/ATTEMPT_ID/`, containing `record.json` and `config.yaml`. Each Runner invocation receives a new attempt ID. This optional recipe section changes record settings:

```yaml
run:
  root_dir: ./experiment_records
  name: baseline
  experiment_id: regression-comparison
  # parent_attempt_id: previous-attempt-id
```

Names and experiment IDs group observations; they are not recipe hashes or execution identities. Set `parent_attempt_id` to the actual earlier attempt when linking a continuation or branch. A checkpoint path alone does not establish lineage. Omit absent optional fields; `parent_attempt_id: null` is invalid. `run: false` disables local records. Invalid literal options fail before project import or model, Trainer and data construction. References, expressions and component values are checked after their normal resolution.

Attempt identity does not make scientific output paths exclusive. Choose fresh directories for separate research stages and concurrent attempts. The diagnostic intentionally shares one directory through a sequential workflow; Compare and Continue rejects reused stage directories.

Programmatic callers receive the native result and may read `runner.last_run_path`, including after process-spawn execution. It is `None` when no record was published. Temporary record callbacks are removed even after stage failure.

## What the evidence means

| Field | Meaning |
|---|---|
| `requested` | Composed source, selected stage arguments and available input-file hashes; expressions remain expressions |
| `observed_start` | Restored progress, selected checkpoint path and optimizer groups/settings observed after restoration |
| `progress`, `metrics` | Updates at training-epoch and fit-validation boundaries |
| `observed_end` | Terminal native progress |
| `checkpoints` | Native ModelCheckpoint references |
| `prediction_destinations` | Configured writer paths and their observed existence at stage end |

Artifact existence does not prove that this attempt wrote its contents. A configuration diff compares intent; an optimizer setting does not prove that an update happened.

Environment entries distinguish distribution versions from actually imported versions/paths and available source Git identities. A source Git identity is reported only when the imported module file is tracked; a wheel in an ignored environment is not attributed to the enclosing application's repository. Git entries include the root and queried directory. These fields are provenance, not complete code/data/environment locks.

## Failures and incomplete records

`running` means no terminal event was recorded, not that a process is alive. Forced termination can leave that state behind. Graceful native exceptions record `failed` or `interrupted`; errors during configuration, import or data preparation before recorder setup may leave no record. Preserve native stderr for those failures.

Recording I/O failures warn and set the recorder diagnostic without replacing a scientific result or its original exception. A record can therefore be incomplete when storage fails. Diff distinguishes a missing field from a literal null.

Opaque Python objects, generators and tensors are described by type and marked non-replayable; recording does not copy or consume them. A snapshot containing descriptors supports inspection, not automatic reconstruction. Even ordinary YAML does not guarantee equal numerical results across environments or unversioned data.

Only rank zero publishes files, while required ranks participate in identity broadcast and metric computation. Source and record publication is atomic. The exercised distributed record profile is two-process CPU/Gloo DDP, not every strategy or storage backend.

## Requested settings and restored results

Full continuation restores optimizer state, including LR and momentum. Read the new requested source and `observed_start` together. A test/predict command's unused optimizer request does not identify how its checkpoint was trained.

With the tested native ModelCheckpoint callback, changing the checkpoint directory skips restoration of old ranking/top-k state. Newly saved checkpoints are ranked in the new attempt, although the old `best_model_path` may remain until another checkpoint is saved. Preserve the preceding attempt's selected model separately and inspect actual metadata. A best path needs its metric, attempt and population context.
