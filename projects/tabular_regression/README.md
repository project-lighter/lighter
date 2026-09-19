# A complete local experiment

This is the download-free diagnostic and reference workflow for Lighter. For the public research walkthrough, start with [Compare and Continue](../experiment_comparison/README.md). This diagnostic fits a linear regressor to a known synthetic relationship, evaluates a concrete checkpoint, exports every held-out prediction with its original ID, and continues training from saved optimizer state. It runs on CPU with no external logger or dataset service.

Install the [current paired sources](../../docs/guides/compatibility.md#install-the-current-source-pair) first. Keep that Python environment active. From the `project-lighter` directory containing the sibling checkouts:

```bash
cd lighter/projects/tabular_regression
```

The working development versions require the matching Sparkwheel source. The [quick start](../../docs/quickstart.md) supplies the full inspection/record command sequence and expected files. Use a new output root for a new diagnostic sequence; the four commands below intentionally share this recipe's output root.

Run the commands yourself:

```bash
lighter fit config.yaml model::optimizer::lr=0.05
lighter test config.yaml --ckpt-path outputs/tabular_regression/checkpoints/last.ckpt --no-verbose
lighter predict config.yaml --ckpt-path outputs/tabular_regression/checkpoints/last.ckpt --no-return-predictions
lighter fit config.yaml --ckpt-path outputs/tabular_regression/checkpoints/last.ckpt trainer::max_epochs=5
```

`python -m lighter` is equivalent to the `lighter` console command. Multiple configuration files are separate arguments; nested overrides use `::`. Each command runs only its requested stage. A fresh process uses an explicit checkpoint path; `best` and `last` shortcuts depend on native Trainer checkpoint context.

For the same workflow with executable checks and saved command logs:

```bash
python workflow.py --output-dir outputs/verified
```

The verifier executes the actual CLI in four separate processes. The fit command overrides the recipe LR from 0.01 to 0.05; native resume restores that saved optimizer LR even though its fresh recipe contains 0.01. It requires 9 training steps after the initial three epochs, all seven prediction IDs exactly once and in order, predictions matching a native Linear loaded from the chosen checkpoint, targets matching the known synthetic relationship, and 15 steps after continuation to five total epochs. It also exercises static inspection and the `runs list`, `runs show` and `runs diff` JSON interfaces, checking four completed attempts and the difference between requested and restored settings. It writes `workflow.json`, command logs, inspection JSON, local attempt records, `predictions.csv` and explicit checkpoints. Checkpoint loading in this verifier uses `weights_only=False` for checkpoints it has just produced itself.

The scientific code in `task.py` contains only the data definition and familiar step methods. The inherited constructor and optimizer hook let Lighter create a fresh optimizer at Lightning's setup point. The YAML explicitly selects the seed, CPU execution, SGD, checkpoint monitor, filenames and CSV columns. `logger: false` disables external logging; automatic callback metrics still drive checkpoint selection. Prediction streaming uses native `--no-return-predictions` and CsvWriter.

The train, validation and test populations use disjoint synthetic sample indices and separate stable identifiers. There is no fitted preprocessing, external download or random split. Validation chooses the monitored best checkpoint, while the commands above intentionally evaluate the last checkpoint; this distinction is explicit. The test set is not used during fitting. IDs include leading zeros, missing-value-like strings, punctuation, Unicode and a newline to verify exact CSV identity preservation.

This is a workflow fixture, not evidence of generalization on a real dataset or a recommended benchmark protocol. Replace its populations and scientific model for your experiment; keep the complete fit/evaluate/export/continue checks. For run inspection and local records, see the [experiment records guide](../../docs/guides/experiment-records.md).
