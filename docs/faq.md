# Frequently Asked Questions

## General

**What is Lighter?**

A configuration-driven deep learning framework built on PyTorch Lightning. Define experiments in YAML instead of writing training code. [Get Started →](tutorials/get-started.md)

**How does Lighter compare to PyTorch Lightning?**

Lighter extends Lightning's `LightningModule` but uses YAML configs instead of Python classes. You get all Lightning features (multi-GPU, callbacks, loggers) plus config-driven simplicity. [Migration Guide →](migration/from-pytorch-lightning.md)

**When should I use Lighter?**

Use Lighter when:

- Running many experiments with different hyperparameters
- Reproducibility and experiment tracking are priorities
- You prefer configuration over code
- You want PyTorch's flexibility with structure

Don't use Lighter when:

- You need ultra-custom training loops
- Rapid architecture prototyping (code first, config later)
- You prefer code-only workflows

**Is Lighter only for medical imaging?**

No. Lighter is task-agnostic and works for any deep learning task: classification, detection, segmentation, NLP, self-supervised learning, etc. [See examples →](how-to/recipes.md)

**What's the performance overhead?**

Minimal (<1%). Config resolution happens once at startup. Training speed is identical to PyTorch Lightning.

## Configuration

**What's the difference between `@`, `%`, and `$`?**

Lighter uses [Sparkwheel](https://project-lighter.github.io/sparkwheel/) for configuration:

| Symbol | Purpose | Example |
|--------|---------|---------|
| `@` | Resolved reference (instantiated object) | `"@system::optimizer"` |
| `%` | Raw reference (unprocessed YAML) | `"%system::metrics::train"` |
| `$` | Evaluate Python expression | `"$0.001 * 2"` |

- `@` gets the final instantiated object after processing
- `%` copies raw YAML config (creates new instance when used with `_target_`)
- `$` evaluates Python code in expressions

[Complete syntax guide →](how-to/configuration.md) | [Sparkwheel docs →](https://project-lighter.github.io/sparkwheel/)

**How do I pass model parameters to the optimizer?**

```yaml
optimizer:
  _target_: torch.optim.Adam
  params: "$@system::model.parameters()"
  lr: 0.001
```

The `$` evaluates Python, `@` gets the resolved model instance, `.parameters()` calls the method.

**Can I use Python code in configs?**

Yes. Use `$` prefix for expressions:

```yaml
optimizer:
  lr: "$0.001 * 2"  # Evaluates to 0.002
```

[Advanced configuration →](how-to/configuration.md#advanced-features)

**How do I add callbacks without replacing existing ones?**

By default, configs merge automatically. Later configs add to earlier ones:

```yaml
# base.yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint

# experiment.yaml (merges automatically)
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.EarlyStopping
# Result: Both ModelCheckpoint AND EarlyStopping

# To replace instead of merge, use =
trainer:
  =callbacks:
    - _target_: pytorch_lightning.callbacks.EarlyStopping
# Result: Only EarlyStopping
```

**How do I remove specific items from lists or dicts?**

Use `~` with path notation or batch syntax:

```yaml
# Delete entire key
trainer:
  ~callbacks: null

# Delete single list item by index
trainer:
  ~callbacks::1: null  # Removes item at index 1

# Delete multiple list items (batch syntax - recommended)
trainer:
  ~callbacks: [1, 3]  # Removes items at indices 1 and 3

# Delete dict keys (batch syntax)
system:
  ~dataloaders: ["train", "test"]  # Removes train and test loaders

# Delete nested dict key (path notation)
system:
  ~model::pretrained: null
```

!!! tip
    For multiple list items, use batch syntax `~key: [indices]` to avoid index shifting issues when doing sequential deletions.

[Merging guide →](how-to/configuration.md#advanced-merging-and)

## Training

**How do I resume training?**

```bash
lighter fit config.yaml args::fit::ckpt_path="checkpoint.ckpt"
```

**How do I use multiple GPUs?**

```bash
lighter fit config.yaml trainer::devices=2 trainer::strategy=ddp
```

[Multi-GPU recipes →](how-to/recipes.md#multi-gpu-ddp)

**How do I freeze layers?**

```yaml
trainer:
  callbacks:
    - _target_: lighter.callbacks.Freezer
      modules: ["backbone.layer1", "backbone.layer2"]
```

[Freezers guide →](how-to/freezers.md)

**Can I use custom training loops?**

Lighter uses Lightning's standard loop. For exotic training logic:

1. Extend System class and override methods
2. Use Lightning directly

Most customizations achievable through callbacks or System extension. [System internals →](design/system.md)

## Design

**Why adapters instead of custom LightningModule?**

Adapters separate data transformation from model logic, making both reusable. Configure transforms in YAML, reuse models across tasks. [Adapter pattern →](how-to/adapters.md)

**What's the difference between stages and modes?**

- **Stages**: CLI commands (fit, validate, test, predict)
- **Modes**: Internal execution contexts (train, val, test, predict)

Example: `lighter fit` executes train + val modes. [Architecture →](design/overview.md#understanding-stages-and-modes)

**How does config pruning work?**

Lighter automatically removes unused components based on stage. `lighter test` removes train/val dataloaders, optimizer, and scheduler. One config works for all stages. [Pruning details →](design/overview.md#automatic-configuration-pruning)

## Troubleshooting

**Training is slow?**

Check:

- Increase `num_workers` in dataloaders
- Enable mixed precision: `trainer::precision="16-mixed"`
- Profile: `trainer::profiler="simple"`

[Performance recipes →](how-to/recipes.md#performance-optimization)

**Loss is NaN?**

Common causes:

1. Learning rate too high → reduce by 10x
2. Gradient explosion → `trainer::gradient_clip_val=1.0`
3. Wrong loss function → verify for your task
4. Bad data → check for inf/nan in inputs

[Full troubleshooting guide →](how-to/troubleshooting.md)

**ModuleNotFoundError: No module named 'project'?**

Ensure:

1. Project path set: `project: ./path`
2. All directories have `__init__.py`

[Project module guide →](how-to/project_module.md)

## Comparisons

**Lighter vs Hydra?**

- **Hydra**: General config framework for any Python app
- **Lighter**: Deep learning-specific with built-in training pipeline

Use Lighter for DL experiments with automatic training loops.

**Lighter vs Ludwig?**

- **Ludwig**: High-level, declarative ML with pre-built flows
- **Lighter**: Mid-level, requires standard PyTorch components

Use Ludwig for no-code ML. Use Lighter when you write custom PyTorch but want config-driven experiments.

**Can I migrate from Lightning to Lighter?**

Yes. Main steps:

1. Convert LightningModule to YAML config
2. Move training_step logic to adapters (if needed)
3. Configure dataloaders in YAML

[Complete migration guide →](migration/from-pytorch-lightning.md)

## Getting Help

| Need | Resource |
|------|----------|
| Getting started | [Tutorials](tutorials/get-started.md) |
| Configuration help | [Configuration Guide](how-to/configuration.md) |
| Common errors | [Troubleshooting](how-to/troubleshooting.md) |
| Examples | [Recipes](how-to/recipes.md) |
| Community | [Discord](https://discord.gg/zJcnp6KrUp) |
| Bug reports | [GitHub Issues](https://github.com/project-lighter/lighter/issues) |
