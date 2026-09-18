# Inspecting and comparing experiments

Runner records each stage attempt locally, including when `trainer.logger: false`. It keeps native Trainer results and exceptions. Use Lightning loggers for richer live dashboards and remote tracking; this record is a portable local summary.

```bash
# Compose and inspect source without importing the configured project or targets.
python -m lighter inspect config.yaml seed=42 --json

python -m lighter fit config.yaml
python -m lighter runs list ./lighter_runs
python -m lighter runs show ./lighter_runs/ATTEMPT_ID
python -m lighter runs diff ./lighter_runs/FIRST ./lighter_runs/SECOND
```

`lighter` and `python -m lighter` expose the same commands. Inspection performs configuration composition, including declared file includes, but does not evaluate configured expressions/imports/constructors. Record operations only read JSON; they do not execute recorded recipes.

## Record location and identity

The default directory is `trainer.default_root_dir/lighter_runs/ATTEMPT_ID/`, containing `record.json` and `config.yaml`. Each Runner invocation receives a new attempt ID. Optional settings:

```yaml
run:
  root_dir: ./experiment_records
  name: baseline
  experiment_id: regression-comparison
  # parent_attempt_id: previous-attempt-id
```

A name or experiment ID groups observations; it is not a recipe hash or an execution identity. Checkpoint continuation and scientific branching both create new attempts. Use `parent_attempt_id` when you want to explicitly link them; a checkpoint path does not automatically establish lineage. `run: false` disables local attempt records. Existing native logger configuration and checkpoint hyperparameters still work.

Programmatic callers receive the native result and can inspect `runner.last_run_path` afterward, including after process-spawn execution. It is `None` when no record was published. Temporary record callbacks are removed even when a stage fails, including when a native model replaces the Trainer callback list.

## What the evidence means

- `requested` preserves composed source, selected stage arguments (including explicit overrides), and available input-file hashes. Expressions remain expressions.
- `observed_start` records native restored progress, selected checkpoint path, and optimizer group settings after restoration. A requested learning rate may differ from the restored one; both are retained. Non-string custom group metadata is described separately.
- `progress` and `metrics` update at training epoch and fit-validation boundaries. `observed_end` records terminal progress. Native loggers remain the choice for batch-level dashboards.
- `checkpoints` contains native ModelCheckpoint references. `prediction_destinations` lists configured Lighter writer paths for prediction, with observed existence at stage end. Existence alone does not prove that an artifact was newly written by this attempt.
- Environment information distinguishes installed distribution versions from actually imported versions/paths and available source Git identity. This is provenance, not a complete environment or dataset lockfile.

Only rank zero publishes files, while all required ranks participate in identity broadcast and metric computation. Source and record files are published atomically. The tested distributed profile is two-process CPU/Gloo DDP; this does not certify every strategy or storage backend.

`running` means no terminal event has been recorded. It is not proof that a process is alive: a forced termination or machine failure can leave a running record. Graceful native exceptions record `failed` or `interrupted`. Records begin when native setup reaches the recorder, so configuration/import/data-preparation errors before that point may have no attempt record. Native stderr and your launcher remain the source for those failures.

Opaque prebuilt Python objects, generators and tensors are described by type and marked non-replayable. They are never copied or consumed merely to record metadata. YAML snapshots with such descriptors support inspection, not automatic reconstruction. Even a plain YAML snapshot does not promise identical numerical results across software, hardware or unversioned data.

Recording I/O failures emit warnings and set the recorder's error diagnostic; they do not replace a scientific result or its original exception. A record can therefore remain incomplete if storage is unavailable. Diff output compares recorded requests and observations, with explicit presence flags to distinguish a missing value from a literal null.
