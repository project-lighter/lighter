# Compare and continue

With the images, starting model and budget fixed, what changes when SGD uses a tenfold larger learning rate? Then, what actually happens when you resume the low-rate run while requesting the larger rate?

This example makes those decisions visible. `task.py` contains the network and scientific steps; `config.yaml` composes the task, native Trainer, data and callbacks. `high_lr.yaml` changes one setting. Lighter constructs the optimizer against the network's actual parameters and records each attempt. The equivalent `native.py` uses ordinary Lightning directly. Both evaluate complete populations and export checkpoint-specific predictions.

The default CPU pilot uses 2,000 training images, 500 validation images and 1,000 final-test images from CIFAR-10. The two-convolution classifier trains for five epochs: 32 updates per epoch, including the 16-image final batch, for 160 updates. SGD has momentum 0.9 and no weight decay. There is no augmentation, dropout, scheduler, accumulation or early stopping. The seeds and per-epoch order are independent of model construction.

## Prepare the data

Install the [current paired sources](../../docs/guides/compatibility.md#install-the-current-source-pair) first and keep that environment active. From the `project-lighter` directory containing the sibling checkouts:

```bash
cd lighter/projects/experiment_comparison
```

Run the remaining commands from this project directory. Download the [official CIFAR-10 Python archive](https://www.cs.toronto.edu/~kriz/cifar.html) into `.cache/cifar-10-python.tar.gz`, then prepare it:

```bash
python data.py prepare --cache-root .cache --output-dir data
```

Preparation verifies the publisher's archive MD5 before reading its Python batch files. It selects 200 training and 50 validation images per class from official training data (seed 1701), and 100 final-test images per class from official test data (seed 1702). It records exact original indices, IDs, labels, SHA-256 hashes and any identical pixels across selected populations. Images become RGB CHW float32 divided by 255 at loading time. Loading never downloads data. A new preparation output directory is required.

Dataset source: Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* (2009); [dataset page and terms](https://www.cs.toronto.edu/~kriz/cifar.html). 

## Inspect, change and fit

```bash
python -m lighter inspect config.yaml --json
python -m lighter inspect config.yaml high_lr.yaml --json
python -m lighter fit config.yaml output_dir=outputs/lr-001
python -m lighter fit config.yaml high_lr.yaml output_dir=outputs/lr-01
```

`inspect` shows composed definitions without constructing the model or running expressions. Compare the two `epochs.csv` files and their `validation.json` records. Selection uses the lowest complete validation cross-entropy, with the earlier checkpoint retained on an exact tie. Measurements sum per-example losses and correct predictions; they do not average unequal batch means. Each `checkpoints.json` names the selected checkpoint and the last checkpoint separately. Its epoch metadata, and the workflow metadata, use one-based epochs. Lightning checkpoint filenames and saved `epoch` indices use zero-based epochs; do not infer the selected epoch from a filename.

Use fresh output directories for every stage. The low-rate fit has a unique attempt in `outputs/lr-001/lighter_runs`:

```bash
python -m lighter runs list outputs/lr-001/lighter_runs --json
```

Keep the final test untouched until the protocol and included seeds are fixed. The first pair is a pilot observation; no LR ranking or model quality is promised. A sensible next experiment follows validation evidence, not a favorable final-test result.

## Evaluate the selected model

Set `BEST` to the exact `selected` path in `outputs/lr-001/checkpoints.json`. These fresh processes explicitly restore that checkpoint. Repeat for the other included LR arm after deciding the final protocol.

```bash
python -m lighter test config.yaml output_dir=outputs/low-test --ckpt-path "$BEST" --no-verbose
python -m lighter predict config.yaml output_dir=outputs/low-predict --ckpt-path "$BEST" --no-return-predictions
```

`low-test/test.json` contains complete loss/count, accuracy and confusion counts. `low-predict/predictions.csv` has one row per manifest image, in that order, with exact `id`, `label`, and `logit_0` through `logit_9`. Lighter's CsvWriter writes batches to disk and native prediction returns no retained prediction list. To explain a particular row, follow its exact ID to the population manifest and its original image index; the named selected checkpoint determines its ten logits.

## Continue with restored momentum

Set `LAST` to the `last` path in the low-rate fit's `checkpoints.json`, and `PARENT` to that fit's attempt ID. Continue into a new output directory:

```bash
python -m lighter fit config.yaml high_lr.yaml output_dir=outputs/low-continued trainer::max_epochs=6 run::parent_attempt_id="$PARENT" --ckpt-path "$LAST"
python -m lighter runs list outputs/low-continued/lighter_runs --json
```

Set `CONTINUED` to the new fit attempt ID, then inspect the requested and observed fields directly:

```bash
python -m lighter runs show "outputs/lr-001/lighter_runs/$PARENT"
python -m lighter runs diff "outputs/lr-001/lighter_runs/$PARENT" "outputs/low-continued/lighter_runs/$CONTINUED"
```

The recipe requests 0.1, but full checkpoint restoration restores the saved optimizer LR **0.01** and its momentum. The sixth epoch extends 160 updates to 192. Read `observed_start.json` and the attempt's requested/observed fields to see both rates. Changing the recipe while restoring the full optimizer is not a learning-rate intervention. Checkpoint selection in the new output directory covers the added epochs of this continuation attempt. The original selected checkpoint remains unchanged. This demonstrates epoch-boundary extension; arbitrary interrupted replay remains outside this example.

## Observed results

The recorded software profile was Python 3.11.14, Torch 2.7.1, torchvision 0.22.1, PyTorch Lightning 2.5.1, TorchMetrics 1.9.0 and NumPy 1.26.4 on macOS 26.3.1 arm64, using Lighter 0.2.0.dev0 and Sparkwheel 0.1.0.dev0. Scientific source hashes are in `results.json`.

All three seeds and both rates were fixed before opening the final test. The table retains every condition from the small fixed CIFAR-10 subset and CPU profile above. Selected epochs are **one-based**.

| Seed | LR | Selected epoch | Validation CE | Test CE | Test accuracy |
| --- | --- | --- | --- | --- | --- |
| 17 | 0.01 | 5 | 2.271701 | 2.273399 | 14.8% |
| 17 | 0.1 | 5 | 2.125193 | 2.121695 | 20.8% |
| 23 | 0.01 | 5 | 2.299727 | 2.298794 | 13.0% |
| 23 | 0.1 | 5 | 2.147750 | 2.162458 | 20.1% |
| 41 | 0.01 | 5 | 2.299483 | 2.299931 | 11.1% |
| 41 | 0.1 | 4 | 2.119862 | 2.135866 | 18.4% |

The matched native Lightning runs produced identical update tensors, gradients, momentum buffers and final logits in this profile. Each final evaluation used the exact validation-selected checkpoint; seed 41 at LR 0.1 selected epoch 4 while its last checkpoint came from epoch 5.

For seed 17, both low-rate continuations requested LR 0.1 and observed restored LR **0.01**. They resumed at update 160, reached update 192, and finished epoch 6 with validation CE **2.251978** and accuracy **16.4%**. Selection covers the added epoch in each new continuation attempt; the original selected checkpoints remain separate.

These are descriptive results for this subset and budget. They do not establish general LR superiority, broader model quality, unfamiliar-user usability or arbitrary interruption recovery. [results.json](results.json) preserves full precision, checkpoint hashes and scientific source identities. Reproduce the process with the [ordinary commands](#inspect-change-and-fit) or the [optional matched workflow](#optional-reproducibility-workflow).

## Optional reproducibility workflow

The ordinary commands above need no control factories or maintainer files. `workflow.py` automates those commands, creates separate stage directories, preserves immutable best/last checkpoint copies and hashes, and links attempts and artifacts in `workflow.json`:

```bash
python workflow.py --backend lighter --data-manifest data/data.json --output-dir outputs/verified-low --seed 17 --lr 0.01 --max-epochs 5
python workflow.py --backend native --data-manifest data/data.json --output-dir outputs/native-low --seed 17 --lr 0.01 --max-epochs 5
```

A fit first inspects the Lighter recipe, measures the starting model on validation data, then trains. The native path records the recipe inspection for comparison context and constructs its own module and Trainer directly. To match independently generated starting tensors and all epoch orders, qualification supplies the optional `--initial-state` and `--order-manifest` pair with its digest sidecar. `--trace-updates` saves each actual update's IDs, loss, parameter tensors, gradients and momentum for an independent reviewer. These verification artifacts do not change the scientific step.

After fixing the protocol, `--evaluate-from` plus `--evaluation final` runs only test/predict from a hashed immutable selected checkpoint. `--continue-from` plus `--parent-attempt-id` restores the immutable last checkpoint and extends training. These modes are exclusive; all artifact hashes are checked before launch. Initial-state controls remain provenance during restoration and cannot overwrite checkpoint weights.

The native baseline has a direct entrypoint too:

```bash
python native.py fit --data-manifest data/data.json --output-dir outputs/direct-native --lr 0.01 --max-epochs 5
```

The comparison concerns this declared CPU profile. The example does not establish unfamiliar-user usability, broader model quality, accelerator behavior or arbitrary recovery semantics. Population totals and output ownership in this example assume one process; scaling requires explicit cross-rank aggregation and output checks beyond changing `trainer::devices`.
