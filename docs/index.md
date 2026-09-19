# Lighter

**Compose an experiment, inspect its definition, then run ordinary Lightning.**

Lighter connects Python scientific code to YAML recipes and command-line overrides. Native Lightning owns execution. You can keep an existing LightningModule or use LighterModule's managed optimizer construction and automatic measurements.

## Get started

[Install the current source pair](guides/compatibility.md#install-the-current-source-pair), then run the [download-free quick start](quickstart.md). The packages in this checkout are unpublished development versions; install both supplied sources together.

| Task | Where to go | What you get |
|---|---|---|
| Define model and data | [Custom code](guides/custom-code.md) | A project with ordinary Python steps and loaders |
| Compose or change settings | [Configuration](guides/configuration.md) | A recipe and small overlays |
| Inspect before execution | [CLI reference](reference/cli.md#inspect-source) | Composed source, optionally JSON |
| Train and choose a checkpoint | [Training workflows](guides/training.md) | A validation-based decision and explicit checkpoint |
| Monitor and compare attempts | [Experiment records](guides/experiment-records.md) | Requested settings, observed state and artifact references |
| Evaluate and export | [Predictions](guides/predictions.md) | Checkpoint-specific values with stable IDs |
| Continue or start a new branch | [Checkpointing](guides/training.md#checkpointing) | An explicit choice about optimizer and progress restoration |

## Choose ownership explicitly

| Approach | Your responsibility | Lighter's contribution |
|---|---|---|
| [Native LightningModule](guides/lightning-module.md) | Steps, logging, optimizer hooks and custom lifecycle | Configuration, CLI and local records |
| [LighterModule](guides/lighter-module.md) | Scientific steps and metric updates | Managed optimizer/scheduler setup and automatic loss/metric logging |

Both use the same recipe system. Changing a target name does not transform arbitrary module code. Custom optimization stays native, and strategy compatibility depends on the model and data policy.

## Learn through a research decision

[Compare and Continue](examples/compare-and-continue.md) compares two learning rates on fixed CIFAR-10 populations with matched native Lightning controls. It separates the validation-selected model from the last checkpoint and shows why a changed recipe LR does not override full-state continuation.

The [example guide](examples/index.md) distinguishes this walkthrough from the installation diagnostic and older integrations. Lighter is general purpose; a new task supplies its own data, objective and correctness checks.

## What a recipe establishes

A recipe describes how to construct a run. It does not freeze external data or dependencies, establish scientific correctness, or prove that a saved artifact came from the intended model. Keep code and data identities, effective runtime settings and the actual checkpoints or predictions alongside it.

See [research practices](guides/best-practices.md), [compatibility](guides/compatibility.md) and [troubleshooting](faq.md). [CLI](reference/cli.md) and the generated [API reference](reference/index.md) provide lookup details.
