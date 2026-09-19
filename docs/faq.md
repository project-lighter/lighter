# Troubleshooting and common questions

Start with the symptom, then follow the authoritative contract. The [quick start](quickstart.md) is a useful complete diagnostic when separating installation issues from a new scientific task.

## Installation or import fails

The current checkout needs the compatible public Sparkwheel source revision. Follow the [paired installation](guides/compatibility.md#install-the-current-source-pair); do not silently downgrade to an older registry dependency.

For `project.*` imports, run from the directory containing `__lighter__.py` and `__init__.py`. A class in `task.py` is `project.task.ClassName`. Normally installed modules keep their ordinary import names. Check the imported file paths if another checkout is unexpectedly selected.

## Inspection passes but fit fails

Inspection composes source without executing configured expressions, targets or scientific steps. Fit performs those operations and native setup. Preserve the actual exception; use a small, separate `fast_dev_run` when real execution is needed. See [debugging](guides/training.md#debugging).

## An optimizer reference is blocked

In the managed path, optimizer/scheduler construction belongs to native setup. An eager callback cannot read `@model::optimizer::lr` during construction. Share a top-level requested scalar, or read the live optimizer in a native runtime hook. [Construction and optimizer ownership](guides/lighter-module.md#construction-and-optimizer-ownership) shows both cases.

## The resumed learning rate ignores my override

Full continuation restores saved optimizer state, including LR and momentum. Read the new requested settings and `observed_start` together. To start a new optimizer policy, create a fresh experiment and load only compatible model weights. See [checkpointing](guides/training.md#checkpointing).

## There is no record, or it still says running

Setup can fail before recording begins. A forced termination may leave a published running record without a terminal event. These statuses are not process-liveness checks. Use stderr and launcher/process information, then [record failure meanings](guides/experiment-records.md#failures-and-incomplete-records).

## My checkpoint monitor is missing

Use the actual emitted metric name. LighterModule's scalar validation loss is `val/loss/epoch`; a native module chooses its own name. Configure the validation loader and step. Turning off an external logger does not disable LighterModule callback measurements, though native stage return dictionaries may differ. See [automatic logging](guides/lighter-module.md#automatic-logging).

## Metrics mix populations or have unexpected weighting

Use independent metric state for independent populations. A `%` definition copy normally constructs separately; an `@` reference shares a resolved object. Managed multi-loader metrics have [specific state limitations](guides/lighter-module.md#separate-evaluation-populations).

Define whether your objective/report averages examples or valid elements. Automatic logging does not infer a masking denominator. Test counts and values, not just labels on a metric name.

## Prediction IDs or memory usage are wrong

Return the original IDs from `predict_step`, configure matching writer keys and read CSV without unwanted type inference. Embedded newlines are part of CSV fields, so physical line counts are not sample counts.

Use `--no-return-predictions` to avoid retaining the complete native list. Other callbacks or your step can still retain data. See [prediction output](guides/predictions.md).

## Do I need to rewrite my LightningModule?

Keep compatible native constructors, hooks and optimization, and compose their inputs in YAML. Lighter does not transform arbitrary scripts or guarantee every strategy. [Native ownership](guides/lightning-module.md) explains migration and public extension points; [compatibility](guides/compatibility.md#exact-validation-profiles) states what has been exercised.

## Can I use multiple GPUs, manual optimization or sharding?

These are native Lightning capabilities whose correctness depends on your module, data and strategy. Read [distributed workflow requirements](guides/training.md#multi-gpu-training) and the managed construction limits before changing devices. Manual optimization owns the actual update algorithm. Ordinary eager-network support is not a sharded-construction claim.

## Where should a new feature live?

Keep task-specific algorithms, masks, sampling and reporting in ordinary Python. Use recipes for composition and reuse. Native callbacks/data hooks are available for lifecycle work. Consult the [public API boundary](guides/lightning-module.md#public-api-and-extension-contracts) before depending on internal helpers.

## Help and citation

Use [GitHub issues](https://github.com/project-lighter/lighter/issues) for reproducible problems or [Discord](https://discord.gg/zJcnp6KrUp) for discussion. Include the exact command, source/version profile, smallest useful recipe and original error.

The citation is *Lighter: Configuration-Driven Deep Learning*, Hadzic et al., Journal of Open Source Software (2025), [doi:10.21105/joss.08101](https://joss.theoj.org/papers/10.21105/joss.08101). The repository README includes the full BibTeX.
