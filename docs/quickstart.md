# Quick Start

Run one complete experiment on CPU with generated data and no external logger. You will change a learning rate, fit a small regressor, evaluate and export a concrete checkpoint, then continue with its saved optimizer state.

## Install and run

First follow [installation](guides/compatibility.md#install-the-current-source-pair). Keep that environment active. From the `project-lighter` directory containing the supplied sibling checkouts:

```bash
cd lighter/projects/tabular_regression
```

The project already contains `__lighter__.py`, `__init__.py`, `task.py` and `config.yaml`. Use a new `outputs/first-experiment` directory for this sequence. If it exists, choose a different name consistently; the commands do not clean previous runs.

### Inspect and change

```bash
python -m lighter inspect config.yaml --json
python -m lighter inspect config.yaml model::optimizer::lr=0.05 --json
```

The first recipe requests SGD LR `0.01`; the override requests `0.05`. Inspection composes definitions without importing the project or evaluating configured expressions. It does not validate a training step by running it.

### Fit, evaluate and export

```bash
python -m lighter fit config.yaml output_dir=outputs/first-experiment model::optimizer::lr=0.05
python -m lighter test config.yaml output_dir=outputs/first-experiment --ckpt-path outputs/first-experiment/checkpoints/last.ckpt --no-verbose
python -m lighter predict config.yaml output_dir=outputs/first-experiment --ckpt-path outputs/first-experiment/checkpoints/last.ckpt --no-return-predictions
```

The fit completes three epochs and nine optimizer updates. The recipe explicitly saves `last.ckpt` and uses a separate validation population to monitor `val/loss/epoch`. These diagnostic commands deliberately evaluate **last**, not the validation-selected best checkpoint.

Prediction exports `outputs/first-experiment/predictions.csv` with columns `id,prediction,target` and seven data records. IDs include `00001`, `NA`, `NULL`, punctuation, Unicode and an embedded newline. Read it with a CSV reader, not by counting physical lines. Prediction streams to the configured writer without retaining the complete native prediction list.

### Continue

```bash
python -m lighter fit config.yaml output_dir=outputs/first-experiment --ckpt-path outputs/first-experiment/checkpoints/last.ckpt trainer::max_epochs=5
```

Five is the **total** epoch limit. The run continues from nine to fifteen optimizer updates. Its recipe requests the default LR `0.01`, but full checkpoint restoration retains saved LR `0.05` and SGD momentum state. An LR override alone is not a fresh fine-tuning experiment.

This small diagnostic intentionally shares an output root across its sequential stages; continuation updates `last.ckpt`. Retain separate checkpoint copies and stage directories for research comparisons, as shown in [Compare and Continue](examples/compare-and-continue.md).

### Inspect the attempts

```bash
python -m lighter runs list outputs/first-experiment/lighter_runs
python -m lighter runs list outputs/first-experiment/lighter_runs --json
```

There are four new attempts: two fits, one test and one predict. In the JSON, find the fit whose `observed_start.global_step` is 0 and the fit whose start step is 9. Set `FIRST` and `CONTINUED` to their actual `attempt_id` values. These values come from your output, not the example text.

For example, assign the variables in your shell, replacing both descriptions with the IDs from your output: `FIRST='your initial fit attempt ID'` and `CONTINUED='your resumed fit attempt ID'`. Then run:

```bash
python -m lighter runs show "outputs/first-experiment/lighter_runs/$FIRST"
python -m lighter runs diff "outputs/first-experiment/lighter_runs/$FIRST" "outputs/first-experiment/lighter_runs/$CONTINUED"
```

`show` and `diff` already emit JSON. Compare the requested source LR with `observed_start.optimizers`, and compare the total epoch limits and restored progress. See [record fields and failure meanings](guides/experiment-records.md).

## Verify the same workflow

From the same project directory, this optional helper runs the four CLI stages in fresh processes under a separate output root:

```bash
python workflow.py --output-dir outputs/verified-first-experiment
```

It checks 9→15 updates, all seven IDs in order, prediction values against the actual checkpoint, restored LR/momentum and four completed records. It saves `workflow.json`, logs, inspection JSON, records and artifacts. This is an installation diagnostic, not evidence of model quality or unfamiliar-user usability.

## Familiar Python, configurable components

The scientific step in `task.py` is ordinary Python:

```python
class RegressionTask(LighterModule):
    def training_step(self, batch, batch_idx):
        prediction = self(batch["x"]).squeeze(-1)
        return self.criterion(prediction, batch["target"])
```

This excerpt uses the imports and other stage methods in the existing project. Its YAML selects a Linear network, MSE criterion and SGD. The inherited constructor builds the network normally; optimizer construction happens at Lightning's setup point. No factory or extra lifecycle hook is needed.

Next, [define your own project](guides/custom-code.md), [compose a recipe](guides/configuration.md), or follow [Compare and Continue](examples/index.md). For custom optimizer ownership, use the [native path](guides/lightning-module.md); for exact managed boundaries, see [construction and optimizer ownership](guides/lighter-module.md#construction-and-optimizer-ownership).
