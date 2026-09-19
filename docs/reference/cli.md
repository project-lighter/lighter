# CLI reference

`lighter` and `python -m lighter` invoke the same CLI. Use the [quick start](../quickstart.md) for a complete executable sequence with known data and outputs.

## Commands

| Command | Purpose |
|---|---|
| `inspect` | Compose and print source without configured execution |
| `fit` | Train with configured validation |
| `validate` | Run validation |
| `test` | Run test evaluation |
| `predict` | Produce native predictions and configured writer artifacts |
| `runs list/show/diff` | Read local records without replaying recipes |

There is no separate export verb: predict plus a writer exports outputs.

## Inspect source

From the diagnostic project:

```bash
python -m lighter inspect config.yaml model::optimizer::lr=0.05 --json
```

`inspect` accepts one or more configuration files/overrides and optional `--json`. Without that flag it emits YAML. Composition may read declared included files, but does not import the configured project/targets, evaluate expressions or construct components. This is source inspection, not runtime validation.

## Stage arguments

General syntax:

```text
lighter STAGE CONFIG [MORE_CONFIGS...] [OVERRIDES...] [OPTIONS...]
```

Pass multiple paths as separate arguments; commas remain literal. Options may appear before, between or after input arguments.

| Option | Stages | Behavior |
|---|---|---|
| `--ckpt-path PATH` | All four | Supply a native checkpoint argument |
| `--weights-only` / `--no-weights-only` | All four | Supply the native deserialization option if the installed Trainer supports it |
| `--verbose` / `--no-verbose` | validate, test | Control native result printing |
| `--return-predictions` / `--no-return-predictions` | predict | Control native prediction retention/return |
| `--help` | Each command | Show its actual parser help |

Underscore aliases such as `--ckpt_path`, `--weights_only` and `--return_predictions` are also supported, including their negative boolean forms. Boolean options are flags: do not append `True` or `False`.

If an option is omitted, YAML stage defaults apply, then the installed native default. Native verbose defaults to true; prediction-return defaults depend on strategy. Unsupported stage arguments fail against the installed Trainer signature. In particular, the reference Lightning 2.5.1 profile does not expose every argument introduced by later Lightning versions.

### lighter fit

Fit invokes training and configured validation; it does not run test. A concrete `--ckpt-path` resumes saved model, optimizer and loop state. `trainer::max_epochs` is the total limit. Changing the recipe LR does not override a restored optimizer automatically.

### lighter validate

Validate invokes the configured validation stage. In a fresh process, name a concrete checkpoint to evaluate trained state. Without one, the command uses the newly configured model.

### lighter test

Test is a separate stage. Reserve its population for final evaluation after the model-selection policy is fixed. As with validate, omitting a checkpoint in a fresh process does not recover a previous fit.

### lighter predict

Predict invokes the model's prediction behavior. Configure a writer to persist artifacts and pass `--no-return-predictions` when the complete native list is unnecessary. Your step supplies IDs and output fields; see [prediction contracts](../guides/predictions.md).

### Checkpoint paths

Native `best` and `last` shortcuts depend on the current Trainer checkpoint context. They are not a cross-process experiment lookup. Use a concrete file path, as in the quick start. The `weights_only` loading option does not request fresh optimizer/loop state during fit; see [checkpointing](../guides/training.md#checkpointing).

## Stage defaults and programmatic precedence

YAML `args::<stage>` supplies native arguments. For example, this is an optional stage-default section for a project with the named checkpoint:

```yaml
args:
  test:
    ckpt_path: checkpoints/selected.ckpt
    verbose: false
```

Explicit CLI options or `Runner.run(..., **kwargs)` override corresponding defaults, including false/null programmatic values. Only the selected stage and non-overridden stage-argument definitions are constructed. The top-level model owns the module; stage arguments cannot replace it. A native DataLoader can be supplied through the appropriate stage argument instead of a DataModule.

Runner accepts a stage string and a list of files, dictionaries or overrides. It returns native results; see [programmatic use](../guides/lightning-module.md#public-api-and-extension-contracts). Fit normally returns None. Evaluation results follow native logging/return semantics.

## Config overrides and merging

`model::optimizer::lr=0.05` is a nested override. Values use Sparkwheel's YAML parsing; quote an entire argument when the shell would otherwise interpret its contents. Later input files compose over earlier ones.

Use `=` and `~` operators for explicit replacement/deletion. Do not assume replacing a callback list behaves like appending one. [Configuration](../guides/configuration.md#merging-configs) explains the source semantics.

## Seeds and project imports

`seed: 42` or `seed=42` seeds Python, NumPy and Torch before project imports and component construction, with native worker seeding enabled. An omitted seed means 0, independent of a previous run's environment.

Seeds must be literal integers in `[0, 4294967295]`. Prebuilt objects cannot be retroactively seeded. Deterministic algorithms are a separate native Trainer setting; a seed alone does not establish equal results across devices or dependency versions.

Runner's project marker convention and installed-package alternative are documented in [custom code](../guides/custom-code.md#the-project-folder-pattern).

## Read and compare records

Syntax uses your actual record root and attempt directories:

```text
lighter runs list ROOT [--json]
lighter runs show ATTEMPT_DIRECTORY_OR_RECORD_JSON
lighter runs diff FIRST_ATTEMPT SECOND_ATTEMPT
```

List emits a small table by default or full records with `--json`. Show and diff already emit JSON and do not accept a `--json` flag. They read files without executing saved definitions.

Use actual IDs from list output; [quick start](../quickstart.md#inspect-the-attempts) demonstrates selection. `running` does not prove process liveness. Records may be absent for early failures; [record meanings](../guides/experiment-records.md) explain the remaining boundaries.

## Troubleshooting and exit status

Parsing and execution errors return a nonzero status. Preserve stderr and the failing command; a missing record does not imply that no user construction occurred. Static inspection and a real `fast_dev_run` answer different questions.

Use `python -m lighter fit --help`, `inspect --help` or `runs --help` for the installed parser. `_LIGHTER_COMPLETE` is not implemented as a shell-completion interface.

`SPARKWHEEL_DEBUG=1`, set before importing Sparkwheel, enables interactive debugging of construction and expressions. It is not a verbose-logging flag: expression debugging returns `None` instead of the normal evaluated result. Leave it unset or `0` for ordinary and unattended runs. The matching sibling guide, `sparkwheel/docs/user-guide/troubleshooting.md`, describes this behavior. Native device/distributed environment variables retain native ownership. See [troubleshooting](../faq.md).
