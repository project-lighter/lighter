# Training workflows

Use one scientific task and a readable recipe to move from a question to checkpoint-specific results. The [quick start](../quickstart.md) is the complete download-free path; [Compare and Continue](../examples/index.md) is the research walkthrough. Command fragments below assume the corresponding project and artifacts exist.

## Basic commands

| Stage | Native purpose |
|---|---|
| `fit` | Train, with configured validation |
| `validate` | Evaluate validation data |
| `test` | Evaluate test data |
| `predict` | Produce prediction outputs and configured writer artifacts |

Fit does not automatically run the test stage. Use `inspect` to compose source before execution and `runs` to read local records afterward. [CLI reference](../reference/cli.md) lists exact arguments and precedence.

## The fit command

Runner composes source, validates/seeds execution inputs, imports the project, constructs components, then calls the native Trainer stage. Construction can execute user code; a successful static inspection is not a successful fit.

Output paths follow the recipe's Trainer, logger and callbacks. No timestamped path is implicit. A local record gives each attempt an ID but does not make arbitrary output directories exclusive.

## Merging configs

Use small overlays for research choices. Multiple paths are separate arguments:

```bash
python -m lighter inspect config.yaml high_lr.yaml --json
```

This example is available in Compare and Continue. Inspect the composition, then use the same inputs for fit with a fresh `output_dir`. Mappings merge recursively; explicit replace/delete operators are explained in [configuration](configuration.md#merging-configs).

## Checkpointing

Three distinct operations need different state decisions:

| Operation | State policy |
|---|---|
| Evaluate a selected model | Restore the exact checkpoint chosen using validation |
| Continue interrupted/finished training | Restore full model, optimizer, scheduler and loop state |
| Start a new optimization experiment | Load compatible model weights into a fresh model/optimizer, without a resume checkpoint |

### Save and select

Configure native `ModelCheckpoint` with the metric name your module actually emits. The diagnostic uses `val/loss/epoch`; a native module using `self.log("val/loss", ...)` has a different name. Use `save_last: true` when you need a last checkpoint.

“Best” depends on a metric, population and attempt. Preserve the selected checkpoint separately from last. Compare and Continue records both paths and their identities. Keep final test data out of the choice.

### Loading checkpoints

In a fresh CLI process, pass a concrete file rather than assuming `best` or `last` discovers a preceding run. For the fitted quick-start diagnostic:

```bash
python -m lighter test config.yaml output_dir=outputs/first-experiment --ckpt-path outputs/first-experiment/checkpoints/last.ckpt --no-verbose
```

This diagnostic intentionally tests last. A research evaluation should use its declared selected model. The native `weights_only` deserialization option, where supported, does not turn a full training resume into model-weights-only initialization.

### Resuming training

```bash
python -m lighter fit config.yaml output_dir=outputs/first-experiment --ckpt-path outputs/first-experiment/checkpoints/last.ckpt trainer::max_epochs=5
```

The epoch limit is total progress, not five additional epochs. Optimizer restoration includes LR and momentum; an overridden recipe LR need not become the effective resumed LR. Read [requested and observed records](experiment-records.md#requested-settings-and-restored-results).

For a research continuation, use a new stage directory and link its parent attempt explicitly. The [public walkthrough](../examples/index.md) shows that sequence. Changing the checkpoint directory can reset native top-k ranking for the new attempt; keep the previous selected checkpoint independently.

### Fine-tuning with fresh optimizer state

Use ordinary Python ownership to construct the fresh model/optimizer and load compatible model weights. Do not pass the old checkpoint as `fit(ckpt_path=...)` if you intend to discard optimizer/progress state. Decide explicitly how changed heads, frozen parameters and optimizer groups should behave. [Native ownership](lightning-module.md) and [freezing](freezing.md) cover those boundaries.

Lighter does not automatically search for a recoverable checkpoint or repair corruption. If loading fails, inspect the original error, verify an allowed earlier checkpoint and report the lost progress. Do not describe a warm start or earlier-state fallback as exact uninterrupted recovery.

## Logging

Use native Lightning loggers for dashboards and external experiment tracking. Set `trainer.logger: false` when you do not need them; Lighter's local attempt records and automatic callback metrics remain available.

The [record guide](experiment-records.md) explains what was requested, what was actually restored and what reached a terminal event. A settings record is not evidence of an optimizer update. Use scientific outputs appropriate to your question.

## Saving predictions

Export is performed by `predict` with a configured writer; it is not a separate CLI verb. The [prediction guide](predictions.md) covers CsvWriter, FileWriter, stable IDs and streaming. Preserve the evaluated checkpoint identity alongside the exact population and output columns.

## Debugging

First inspect the source. If construction or behavior needs checking, a small actual run can help:

```bash
python -m lighter fit config.yaml trainer::fast_dev_run=true
```

This executes imports, constructors and real batches, but native fast-dev mode suppresses logging and checkpoint saving. Use an ordinary bounded run to diagnose a logger or checkpoint callback. Use a separate output root when diagnosing a populated project. Start with CPU/one device and the small complete diagnostic when isolating installation or lifecycle problems. Native tracebacks remain the source for failures before records begin.

Check metric names, tensor shapes/dtypes, sample IDs and actual update counts before escalating to a full experiment. [Troubleshooting](../faq.md) maps common symptoms to the relevant contract.

## Multi-GPU training

Lighter passes native Trainer strategy settings through; a recipe change does not establish correct distributed science. Before scaling, define sampling, cross-rank metric reduction, shared storage and output ownership. Global batch size includes per-rank batch size, ranks and accumulation; unequal partial windows and masked objectives need deliberate weighting.

Use native Lightning's [strategy documentation](https://lightning.ai/docs/pytorch/stable/extensions/strategy.html) and the [qualified-profile limits](compatibility.md#exact-validation-profiles). Freezing has [specific DDP requirements](freezing.md). Sharded model construction and arbitrary distributed checkpoint continuation are not implied by ordinary eager-network support.

## Research practices

Treat scheduler units, clipping, accumulation, precision and tuning as algorithm choices. LR finders expect a native module's tuning attribute and optimizer hook; a managed YAML optimizer field is not automatically that attribute. Consult current native APIs for advanced choices, then check the actual scientific update.

A useful run ends with an interpretable result and its limitations. Use the [research checklist](best-practices.md) and compare all declared conditions rather than only the most favorable final result.
