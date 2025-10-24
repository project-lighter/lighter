# Frequently Asked Questions

## General

### What is Lighter?
A configuration-driven deep learning framework built on PyTorch Lightning. Define experiments in YAML instead of writing training code.

### How does Lighter compare to PyTorch Lightning?
Lighter uses PyTorch Lightning's Trainer but adds configuration-driven experiments. Define everything in YAML instead of LightningModule classes, with even more boilerplate taken care of for you. See [Migration Guide](migration/from-pytorch-lightning.md).

### When should I use Lighter?
- Running many experiments with different configurations
- Reproducibility and experiment tracking are priorities
- You prefer configuration over code
- You want a lightweight, flexible framework

### Is Lighter only for medical imaging?
No! While it integrates with MONAI, Lighter works for any task: classification, detection, segmentation, NLP, self-supervised learning, etc.

## Configuration

### What's the difference between `@`, `%`, and `$`?

- `@` = Reference to instantiated Python object
- `%` = Reference to YAML value (textual copy)
- `$` = Evaluate as Python expression

See [Configuration Guide](how-to/configure.md) for details.

### Can I use Python code in configs?

Yes! This is called *evaluation*—think `eval()` in Python. Use `$` prefix:
```yaml
optimizer:
  lr: "$0.001 * 2"  # Evaluates to 0.002
```

Another example is passing model's parameters to the optimizer. To do this, you need to:

1. Get the model definition via `system#model`
2. Instantiate it using `@`
3. Evaluate its `.parameters()` method using `$`

```yaml
optimizer:
  params: "$@system#model.parameters()"
```

### How do I debug config errors?
See [Troubleshooting Guide](how-to/troubleshooting.md).

## Training

### How do I resume training?

```bash
lighter fit config.yaml --args#fit#ckpt_path="path/to/checkpoint.ckpt"
```

### How do I use multiple GPUs?

```bash
lighter fit config.yaml --trainer#devices=2 --trainer#strategy=ddp
```
See [PyTorch Lightning distributed docs](https://lightning.ai/docs/pytorch/stable/common/trainer.html#devices).

### Why is my training slow?
Check:

- Increase `num_workers` in dataloaders
- Enable mixed precision (`--trainer#precision="16-mixed"`)
- Use profiler (`--trainer#profiler="simple"`)

See [PyTorch Lightning performance docs](https://lightning.ai/docs/pytorch/stable/tuning/profiler.html).

## Troubleshooting

### ModuleNotFoundError: No module named 'project'

Ensure:

1. Project path is set correctly: `project: <PATH_TO_YOUR_PROJECT>`
2. All module directories have `__init__.py`

See [Troubleshooting Guide](how-to/troubleshooting.md).

## Getting Help

- [Troubleshooting Guide](how-to/troubleshooting.md)
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)
- [Discord](https://discord.gg/zJcnp6KrUp)
- [GitHub Issues](https://github.com/project-lighter/lighter/issues)
