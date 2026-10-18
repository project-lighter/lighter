---
title: FAQ
---

# Frequently Asked Questions

Quick answers to common questions.

## What is Lighter?

Lighter is a YAML configuration layer for PyTorch Lightning experiments.

**You write:** Standard PyTorch Lightning code (LightningModule, datasets, etc.)

**Lighter provides:** YAML configs, CLI overrides, and each run's config saved with its logs and checkpoints

**Result:** Reproducible experiments without hardcoded hyperparameters.

## Do I need to rewrite my LightningModule?

No! Use it directly:

```yaml
model:
  _target_: project.model.MyLightningModule  # Your existing code, in model.py next to __lighter__.py
  learning_rate: 0.001
```

No changes to your Python code required.

## When should I use LighterModule vs my own LightningModule?

**Use LightningModule when:**

- Migrating existing projects
- Need custom training logic
- Want full control
- Team knows Lightning well

**Use LighterModule when:**

- Starting new projects
- Want less boilerplate
- Standard workflows (classification, segmentation, etc.)
- Config-driven everything

Both are equally supported and give you YAML configs + CLI overrides.

## How do I use my custom models and datasets?

Two steps:

1. Add an empty `__lighter__.py` to your project folder
2. Reference your code as `project.module.ClassName` in config

```yaml
model:
  network:
    _target_: project.models.CustomNet
    num_classes: 10
```

[Full guide →](guides/custom-code.md)

## How do I override config from CLI?

Use `::` to navigate config paths:

```bash
# Single override
lighter fit config.yaml model::optimizer::lr=0.01

# Multiple overrides
lighter fit config.yaml \
  model::optimizer::lr=0.01 \
  trainer::max_epochs=100 \
  data::train_dataloader::batch_size=64
```

No file editing needed!

## What's the difference between `@` and `%`?

- **`@`** = Resolved reference (gets the instantiated Python object)
- **`%`** = Raw reference (copies the YAML config to create new instance)

**Critical:** Always use `%` for metrics:

```yaml
# ❌ WRONG - Shared instance pollutes metrics
val_metrics: "@model::train_metrics"

# ✅ CORRECT - New instance for validation
val_metrics: "%model::train_metrics"
```

Use `@` to pass the same object or value, e.g. `out_features: "@num_classes"`.

## Can I use multiple GPUs?

Yes! Same as PyTorch Lightning:

```yaml
trainer:
  devices: 4  # Use 4 GPUs
  strategy: ddp  # Distributed Data Parallel
```

Or use all available GPUs:

```yaml
trainer:
  devices: -1  # All GPUs
  strategy: ddp
```

[Multi-GPU guide →](guides/training.md#multi-gpu-training)

## How do I debug config errors?

### 1. Start Simple

Run one batch to catch errors fast:

```bash
lighter fit config.yaml trainer::fast_dev_run=true
```

### 2. Check Common Issues

**Import errors** - Check:

- `__lighter__.py` exists in your project folder (or a parent folder of your config)
- Your `_target_` starts with `project.`

Print the composed config without building anything:

```bash
lighter fit config.yaml --print_config
```

### 3. Validate Config Syntax

Use quotes for expressions:

```yaml
# ❌ WRONG
lr: $0.001 * 2

# ✅ CORRECT
lr: "$0.001 * 2"
```

[Full troubleshooting →](guides/training.md#debugging)

## How do I save predictions?

Use Writers:

```yaml
trainer:
  callbacks:
    - _target_: lighter.callbacks.CsvWriter
      path: predictions.csv
      keys: [prediction, probability]
```

In your module:

```python
def predict_step(self, batch, batch_idx):
    x, _ = batch
    pred = self(x)

    return {
        "prediction": pred.argmax(dim=1),
        "probability": pred.max(dim=1).values,
    }
```

Results saved to `predictions.csv`.

[Writers guide →](guides/training.md#saving-predictions)

## Can I merge multiple configs?

Yes! Pass several files, separated by spaces:

```bash
lighter fit base.yaml experiment.yaml
```

Later files override earlier ones. Use this for:

- Base config + experiment-specific overrides
- Shared settings + dataset configs
- Default values + hyperparameter sweeps

[Config merging →](guides/training.md#merging-configs)

## How does Lighter compare to Hydra?

Similar concept, different focus:

**Hydra:**

- General-purpose Python app configuration
- Works with any codebase
- Manual integration

**Lighter:**

- Specialized for PyTorch Lightning
- Built-in CLI commands (`fit`, `validate`, `test`, `predict`)
- Automatic object instantiation
- No sweeps, launchers or config groups of its own: stack config files, and use Lightning's run folders and your logger

If you're doing deep learning with Lightning, Lighter is simpler. If you need general Python app config, use Hydra.

## Where can I get help?

- **Discord**: [discord.gg/zJcnp6KrUp](https://discord.gg/zJcnp6KrUp) - Community support
- **GitHub**: [github.com/project-lighter/lighter](https://github.com/project-lighter/lighter) - Issues
- **Docs**: You're here! Check the guides.

## How do I cite Lighter?

```bibtex
@article{lighter,
    doi = {10.21105/joss.08101},
    year = {2025}, publisher = {The Open Journal}, volume = {10}, number = {111}, pages = {8101},
    author = {Hadzic, Ibrahim and Pai, Suraj and Bressem, Keno and Foldyna, Borek and Aerts, Hugo JWL},
    title = {Lighter: Configuration-Driven Deep Learning},
    journal = {Journal of Open Source Software}
}
```

[Paper →](https://joss.theoj.org/papers/10.21105/joss.08101)

## Can I contribute?

Yes! Lighter is open source:

- **Report bugs**: [GitHub Issues](https://github.com/project-lighter/lighter/issues)
- **Request features**: [GitHub Issues](https://github.com/project-lighter/lighter/issues)
- **Contribute code**: Submit PRs

See [CONTRIBUTING.md](https://github.com/project-lighter/lighter/blob/main/CONTRIBUTING.md) for guidelines.

## More Questions?

Check out:

- [Quick Start](quickstart.md) - Get started in 10 minutes
- [Configuration Guide](guides/configuration.md) - Master the syntax
- [Training Guide](guides/training.md) - Run experiments
- [Best Practices](guides/best-practices.md) - Production patterns
