---
title: FAQ
---

# Frequently Asked Questions

Quick answers to common questions.

## What is Lighter?

Lighter is a YAML configuration layer for PyTorch Lightning experiments.

**You write:** Standard PyTorch Lightning code (LightningModule, datasets, etc.)

**Lighter provides:** YAML configs, CLI overrides, experiment tracking

**Result:** Reproducible experiments without hardcoded hyperparameters.

## Do I need to rewrite my LightningModule?

No! Use it directly:

```yaml
model:
  _target_: my_project.MyLightningModule  # Your existing code
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

Three steps:

1. Add `__lighter__.py` to your project root (can be empty)
2. Ensure all directories have `__init__.py`
3. Reference as `project.module.ClassName` in config

```yaml
model:
  network:
    _target_: my_project.models.CustomNet
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

Use `@` for everything else (optimizer, scheduler, network):

```yaml
# ✅ CORRECT - Pass actual object
optimizer:
  params: "$@model::network.parameters()"
```

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

- `__lighter__.py` exists in project root
- All directories have `__init__.py`
- Running `lighter` from directory with `__lighter__.py`

**Attribute errors** - Using `::` for Python methods?

```yaml
# ❌ WRONG
params: "$@model::network::parameters()"

# ✅ CORRECT
params: "$@model::network.parameters()"
```

Remember: `::` for config, `.` for Python.

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
    - _target_: lighter.callbacks.CSVWriter
      write_interval: batch
```

In your module:

```python
def predict_step(self, batch, batch_idx):
    x = batch
    pred = self(x)

    return {
        "prediction": pred.argmax(dim=1),
        "probability": pred.max(dim=1).values,
    }
```

Results saved to `predictions.csv`.

[Writers guide →](guides/training.md#saving-predictions)

## Can I merge multiple configs?

Yes! Separate comma-separated paths:

```bash
lighter fit base.yaml,experiment.yaml
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
- Built-in CLI commands (`fit`, `test`, `predict`)
- Automatic object instantiation
- Zero-config experiment tracking

If you're doing deep learning with Lightning, Lighter is simpler. If you need general Python app config, use Hydra.

## Where can I get help?

- **Discord**: [discord.gg/zJcnp6KrUp](https://discord.gg/zJcnp6KrUp) - Community support
- **GitHub**: [github.com/project-lighter/lighter](https://github.com/project-lighter/lighter) - Issues and discussions
- **Docs**: You're here! Check the guides.

## How do I cite Lighter?

```bibtex
@article{lighter2024,
  title={Lighter: A lightweight deep learning framework for rapid experimentation},
  author={...},
  journal={Journal of Open Source Software},
  year={2024},
  doi={10.21105/joss.08101}
}
```

[Paper →](https://joss.theoj.org/papers/10.21105/joss.08101)

## Can I contribute?

Yes! Lighter is open source:

- **Report bugs**: [GitHub Issues](https://github.com/project-lighter/lighter/issues)
- **Request features**: [GitHub Discussions](https://github.com/project-lighter/lighter/discussions)
- **Contribute code**: Submit PRs

See [CONTRIBUTING.md](https://github.com/project-lighter/lighter/blob/main/CONTRIBUTING.md) for guidelines.

## More Questions?

Check out:

- [Quick Start](quickstart.md) - Get started in 10 minutes
- [Configuration Guide](guides/configuration.md) - Master the syntax
- [Training Guide](guides/training.md) - Run experiments
- [Best Practices](guides/best-practices.md) - Production patterns
