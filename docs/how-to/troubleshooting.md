# Troubleshooting

Common errors and solutions when using Lighter.

## Configuration Errors

### ModuleNotFoundError: No module named 'project'

**Cause:** Missing `__init__.py` files or incorrect project path

**Solution:**
```yaml
# In your config.yaml
project: ./my_project  # Ensure path is correct

# Ensure all module directories have __init__.py:
my_project/
├── __init__.py       # Required!
├── models/
│   ├── __init__.py   # Required!
│   └── my_model.py
```

### Config Reference Errors

**Wrong:** `"$@system#model#parameters()"` - Using `#` for attributes
**Correct:** `"$@system#model.parameters()"` - Use `.` for Python attributes

**Wrong:** Circular references
```yaml
model:
  lr: "@system#optimizer#lr"  # Circular!
optimizer:
  lr: "@system#model.lr"      # Circular!
```

**Correct:** Use `vars` section
```yaml
vars:
  lr: 0.001
model:
  lr: "%vars#lr"
optimizer:
  lr: "%vars#lr"
```

### YAML Syntax Errors

Common mistakes:
- Missing colons after keys
- Inconsistent indentation (use spaces, not tabs)
- Missing quotes around values with special characters
- Missing values (like the `roi_size` example in inferers)

## Training Issues

### CUDA Out of Memory

**Solutions:**
```bash
# Reduce batch size
lighter fit config.yaml --system#dataloaders#train#batch_size=8

# Enable gradient accumulation
lighter fit config.yaml --trainer#accumulate_grad_batches=4

# Use mixed precision
lighter fit config.yaml --trainer#precision="16-mixed"
```

For distributed strategies, see [PyTorch Lightning docs](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html).

### Loss is NaN

**Check:**
1. Learning rate too high → Reduce by 10x
2. Missing data normalization → Add transforms
3. Wrong loss function for task → Verify criterion
4. Gradient explosion → Add gradient clipping in Trainer config

### Slow Training

**Optimize:**
```yaml
system:
  dataloaders:
    train:
      num_workers: 8      # Increase for faster data loading
      pin_memory: true    # For GPU training
      persistent_workers: true  # Reduce worker startup overhead
```

For profiling and optimization, see [PyTorch Lightning performance docs](https://lightning.ai/docs/pytorch/stable/tuning/profiler.html).

## Debugging Strategies

### Quick Testing
```bash
# Test with 2 batches only
lighter fit config.yaml --trainer#fast_dev_run=2
```

### Debug Config Values
```yaml
# Print values during config resolution
optimizer:
  lr: "$print('LR:', 0.001) or 0.001"
```

### Check Flow Outputs
Temporarily add print transforms in flows:
```yaml
flows:
  train:
    criterion:
      pred: "$lambda x: print('Pred shape:', x.shape) or x"
```

## Getting Help

1. Search this documentation
2. Check [FAQ](../faq.md)
3. Review [PyTorch Lightning docs](https://lightning.ai/docs/pytorch/stable/) for Trainer issues
4. Join [Discord](https://discord.gg/zJcnp6KrUp)
5. Open [GitHub issue](https://github.com/project-lighter/lighter/issues)
