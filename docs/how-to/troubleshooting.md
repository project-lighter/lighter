# Troubleshooting

Comprehensive guide to debugging errors and issues in Lighter. Includes actual error messages and step-by-step solutions.

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

**Wrong:** `"$@system::model::parameters()"` - Using `::` for attributes
**Correct:** `"$@system::model.parameters()"` - Use `.` for Python attributes

**Wrong:** Circular references
```yaml
model:
  lr: "@system::optimizer::lr"  # Circular!
optimizer:
  lr: "@system::model.lr"      # Circular!
```

**Correct:** Use `vars` section
```yaml
vars:
  lr: 0.001
model:
  lr: "%vars::lr"
optimizer:
  lr: "%vars::lr"
```

### YAML Syntax Errors

Common mistakes:
- Missing colons after keys
- Inconsistent indentation (use spaces, not tabs)
- Missing quotes around values with special characters
- Missing values (like the `roi_size` example in inferers)

### Sparkwheel Validation Errors

Lighter uses Sparkwheel for config validation. Here's how to interpret validation errors:

**Error Example:**
```
ValueError: Configuration validation failed:
system.model: Missing required field '_target_'
system.optimizer.lr: Expected float, got str
```

**Solution:**
Check the schema in `src/lighter/engine/schema.py` or add missing fields:
```yaml
system:
  model:
    _target_: torch.nn.Linear  # Must have _target_
  optimizer:
    lr: 0.001  # Not "0.001" (string)
```

**Error: Missing required component**
```
ValueError: Configuration validation failed:
system.optimizer: Required field missing
```

**Solution:** Lighter requires certain components depending on the stage:
- FIT stage: model, optimizer, criterion, train dataloader required
- VALIDATE stage: model, criterion, val dataloader required
- TEST stage: model, test dataloader required
- PREDICT stage: model, predict dataloader required

### Reference Resolution Errors

**Error: Reference not found**
```
KeyError: 'Could not resolve reference @system::modell'
```

**Solution:** Typo in reference path. Check spelling:
```yaml
optimizer:
  params: "$@system::model.parameters()"  # Not 'modell'
```

**Error: Attribute not found**
```
AttributeError: 'ResNet' object has no attribute 'paramters'
```

**Solution:** Typo in method name:
```yaml
optimizer:
  params: "$@system::model.parameters()"  # Not 'paramters'
```

**Error: Using :: for Python attributes**
```
# Wrong
params: "$@system::model::parameters()"

# Correct
params: "$@system::model.parameters()"
```

**Rule**: Use `::` for config navigation, `.` for Python attributes

## Training Issues

### CUDA Out of Memory

**Solutions:**
```bash
# Reduce batch size
lighter fit config.yaml system::dataloaders::train::batch_size=8

# Enable gradient accumulation
lighter fit config.yaml trainer::accumulate_grad_batches=4

# Use mixed precision
lighter fit config.yaml trainer::precision="16-mixed"
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

### Metrics Not Computing

**Error:** No metrics logged to TensorBoard/W&B

**Cause 1:** Metrics not defined for the mode
```yaml
# Wrong: No val metrics
system:
  metrics:
    train:
      - _target_: torchmetrics.Accuracy
```

**Solution:** Add metrics for each mode
```yaml
system:
  metrics:
    train:
      - _target_: torchmetrics.Accuracy
        task: multiclass
        num_classes: 10
    val: "%system::metrics::train"  # Reuse train config
```

**Cause 2:** MetricsAdapter argument mismatch
```yaml
# Wrong: Metric expects 'preds', but adapter uses default (positional)
system:
  adapters:
    train:
      metrics:
        _target_: lighter.adapters.MetricsAdapter
        # Missing pred_argument and target_argument
```

**Solution:** Match metric signature
```yaml
system:
  adapters:
    train:
      metrics:
        _target_: lighter.adapters.MetricsAdapter
        pred_argument: "preds"
        target_argument: "target"
```

### Model Not Learning (Loss Not Decreasing)

**Check 1: Are gradients flowing?**

Add to config temporarily:
```yaml
_requires_:
  - "$import torch"

system:
  model:
    _target_: MyModel
    # Check grad flow after first batch
    _debug: "$print('Has grad:', next(iter(@system::model.parameters())).grad is not None)"
```

**Check 2: Is optimizer updating weights?**

Enable optimizer stats logging (automatic in Lighter):
```yaml
trainer:
  callbacks:
    - _target_: pytorch_lightning.callbacks.LearningRateMonitor
      logging_interval: step
```

Check `train/lr` in logs. If it's not changing with a scheduler, scheduler may be misconfigured.

**Check 3: Is loss function appropriate?**

- Classification: Use `CrossEntropyLoss` (takes logits, not probabilities)
- Regression: Use `MSELoss`
- Binary: Use `BCEWithLogitsLoss` (takes logits)

**Common mistake:**
```yaml
# Wrong: Applying softmax before CrossEntropyLoss
system:
  adapters:
    train:
      criterion:
        pred_transforms:
          - _target_: torch.nn.functional.softmax  # ❌ Don't do this!
  criterion:
    _target_: torch.nn.CrossEntropyLoss
```

**Correct:**
```yaml
# CrossEntropyLoss applies softmax internally
system:
  criterion:
    _target_: torch.nn.CrossEntropyLoss  # No softmax needed
```

## Distributed Training Issues (DDP)

### Error: Address already in use

**Error:**
```
RuntimeError: Address already in use
```

**Cause:** Previous DDP process didn't terminate cleanly

**Solution:**
```bash
# Find and kill lingering processes
ps aux | grep python
kill -9 <PID>

# Or restart your terminal/jupyter kernel
```

### File Writing Conflicts

**Error:** Multiple processes writing to same file causing corruption

**Cause:** Writers in predict mode running on all GPUs

**Solution:** Use rank-zero only for file operations
```python
# In custom callback
from pytorch_lightning.utilities import rank_zero_only

class MyWriter(Callback):
    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only rank 0 writes
        save_predictions(outputs)
```

Lighter's built-in Writers handle this automatically.

### Metrics Aggregation Issues

**Error:** Metrics values differ across GPUs

**Solution:** Lighter automatically sets `sync_dist=True` for epoch-level metrics. For manual logging:
```python
self.log("custom_metric", value, on_epoch=True, sync_dist=True)
```

### Different Behavior on Single vs Multi-GPU

**Cause:** Batch normalization or dropout behaving differently

**Solution:** Use `sync_batchnorm` for BN:
```yaml
trainer:
  strategy: ddp
  sync_batchnorm: true  # Synchronize BN statistics
```

## Adapter Debugging

### Inspecting Tensor Shapes

Add print transforms to see tensor shapes at each stage:

```yaml
system:
  adapters:
    train:
      batch:
        _target_: lighter.adapters.BatchAdapter
        input_accessor: "image"
        target_accessor: "mask"
        # Debug: Print shapes after extraction
        input_transforms:
          - "$lambda x: print(f'Input shape: {x.shape}') or x"
        target_transforms:
          - "$lambda x: print(f'Target shape: {x.shape}') or x"

      criterion:
        _target_: lighter.adapters.CriterionAdapter
        pred_transforms:
          - "$lambda x: print(f'Pred before softmax: {x.shape}') or x"
          - _target_: torch.nn.functional.softmax
            dim: 1
          - "$lambda x: print(f'Pred after softmax: {x.shape}') or x"
```

### Common Adapter Errors

**Error: Wrong argument order**
```
TypeError: forward() got an unexpected keyword argument 'prediction'
```

**Solution:** Check loss function signature and match adapter:
```python
# If loss expects loss(pred, target)
def my_loss(pred, target):
    ...

# Adapter config (default is correct)
system:
  adapters:
    train:
      criterion:
        _target_: lighter.adapters.CriterionAdapter
        pred_argument: 0
        target_argument: 1
```

**Error: KeyError in batch**
```
KeyError: 'image'
```

**Solution:** Check dataset output:
```python
# Temporarily add to dataset
def __getitem__(self, idx):
    batch = {...}
    print(f"Batch keys: {batch.keys()}")  # Debug
    return batch
```

Then match `input_accessor` to actual key.

## Performance Profiling

### Identify Bottlenecks

Use PyTorch Lightning profiler:

```yaml
trainer:
  profiler: simple  # or 'advanced', 'pytorch'
  max_epochs: 1
  limit_train_batches: 100
```

Output shows time spent in each method:
```
╒═══════════════════════╤═════════╤═════════╕
│ Action                │ Mean    │ Total   │
╞═══════════════════════╪═════════╪═════════╡
│ run_training_batch    │ 0.105   │ 10.5    │
│ get_train_batch       │ 0.095   │ 9.5     │
│ training_step         │ 0.008   │ 0.8     │
╘═══════════════════════╧═════════╧═════════╛
```

If `get_train_batch` is slow: Increase `num_workers` in dataloader

If `training_step` is slow: Profile model forward/loss computation

### Data Loading Bottleneck

**Symptom:** GPU underutilized, low GPU usage

**Solution:**
```yaml
system:
  dataloaders:
    train:
      num_workers: 8          # Increase (up to CPU cores)
      prefetch_factor: 4      # Prefetch more batches
      persistent_workers: true # Reuse workers
      pin_memory: true        # Faster GPU transfer
```

**Test different num_workers:**
```bash
for i in 2 4 8 16; do
  echo "Testing num_workers=$i"
  lighter fit config.yaml \
    system::dataloaders::train::num_workers=$i \
    trainer::limit_train_batches=100 \
    trainer::max_epochs=1
done
```

### Model Bottleneck

**Symptom:** High GPU usage, slow batches

**Solutions:**

1. **Mixed precision training:**
```yaml
trainer:
  precision: "16-mixed"  # ~2x speedup, less memory
```

2. **Gradient accumulation (simulate larger batch):**
```yaml
trainer:
  accumulate_grad_batches: 4  # Update every 4 batches

system:
  dataloaders:
    train:
      batch_size: 8  # Effective: 8 * 4 = 32
```

3. **Compile model (PyTorch 2.0+):**
```yaml
_requires_:
  - "$import torch"

system:
  model:
    _target_: torchvision.models.resnet50
    # Compile for faster execution
    _post_init_: "$lambda m: torch.compile(m)"
```

## Memory Optimization

### Beyond Reducing Batch Size

**Strategy 1: Gradient Checkpointing**

Trade compute for memory (recompute activations during backward):

```python title="my_project/model.py"
from torch.utils.checkpoint import checkpoint_sequential

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[
            ResidualBlock() for _ in range(50)
        ])

    def forward(self, x):
        # Checkpoint every 10 layers
        return checkpoint_sequential(self.layers, 10, x)
```

**Strategy 2: CPU Offloading**

Move intermediate results to CPU:

```yaml
system:
  adapters:
    val:
      logging:
        _target_: lighter.adapters.LoggingAdapter
        # Move to CPU for callbacks
        pred_transforms:
          - "$lambda x: x.cpu()"
```

**Strategy 3: Clear Cache Periodically**

```python title="my_project/custom_system.py"
from lighter.system import System
import torch

class MemoryEfficientSystem(System):
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Clear cache every 100 batches
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
```

**Strategy 4: Use Smaller Precision**

```yaml
trainer:
  precision: "bf16-mixed"  # BFloat16 (less memory than fp16 in some cases)
```

## Debugging Strategies

### Quick Testing
```bash
# Test with 2 batches only
lighter fit config.yaml trainer::fast_dev_run=2
```

### Debug Config Values
```yaml
# Print values during config resolution
optimizer:
  lr: "$print('LR:', 0.001) or 0.001"
```

### Check Adapter Outputs
Temporarily add print transforms in adapters:
```yaml
adapters:
  train:
    criterion:
      pred_transforms:
        - "$lambda x: print('Pred shape:', x.shape) or x"
```

### Debug Mode Checklist

When encountering an error:

1. **Test with fast_dev_run:**
   ```bash
   lighter fit config.yaml trainer::fast_dev_run=true
   ```

2. **Verify config resolves:**
   ```yaml
   _requires_:
     - "$import sys"
   vars:
     _debug: "$print('Config loaded successfully', file=sys.stderr)"
   ```

3. **Check tensor shapes:**
   Add print transforms in adapters (see Adapter Debugging above)

4. **Isolate the component:**
   Test individual components in Python:
   ```python
   from sparkwheel import Config
   config = Config.from_file("config.yaml")
   model = config.resolve("system::model")
   print(model)  # Does it instantiate correctly?
   ```

5. **Check logs:**
   Look for warnings/errors in the console output

## Common Error Messages Reference

### TypeError: unhashable type: 'dict'

**Cause:** Passing a dict where Lighter expects a hashable key

**Common scenario:** Using dict-based batch in metric that expects tensors

**Solution:** Use MetricsAdapter to extract tensors:
```yaml
system:
  adapters:
    train:
      metrics:
        pred_transforms:
          - "$lambda x: x['logits']"  # Extract tensor from dict
```

### RuntimeError: Expected all tensors to be on the same device

**Cause:** Model on GPU but data on CPU (or vice versa)

**Solution:** Lighter handles this automatically. If you see this:
- Check custom transforms aren't moving data
- Ensure model is properly registered in System
- For manual operations, use `self.device`:
  ```python
  def custom_operation(self):
      tensor = torch.tensor([1, 2, 3]).to(self.device)
  ```

### ValueError: The loss dictionary must include a 'total' key

**Cause:** Dict-based loss missing required 'total' key

**Solution:**
```python
def my_criterion(pred, target):
    loss1 = ...
    loss2 = ...
    return {
        "total": loss1 + loss2,  # Required!
        "classification": loss1,
        "segmentation": loss2,
    }
```

### OSError: [Errno 24] Too many open files

**Cause:** Too many num_workers, system limit reached

**Solution:**
```bash
# Temporary fix (macOS/Linux)
ulimit -n 4096

# Or reduce num_workers
lighter fit config.yaml system::dataloaders::train::num_workers=4
```

## Real Error Examples from Users

### Example 1: Circular Reference

**Error:**
```
RecursionError: maximum recursion depth exceeded
```

**User's config:**
```yaml
system:
  model:
    _target_: MyModel
    optimizer: "@system::optimizer"  # ❌ Circular!
  optimizer:
    _target_: torch.optim.Adam
    params: "$@system::model.parameters()"  # ❌ Circular!
```

**Fix:**
```yaml
vars:
  lr: 0.001

system:
  model:
    _target_: MyModel
    lr: "%vars::lr"  # Use var instead
  optimizer:
    _target_: torch.optim.Adam
    params: "$@system::model.parameters()"
    lr: "%vars::lr"
```

### Example 2: Wrong Loss Function for Task

**Error:** Loss is negative or NaN

**User's config:**
```yaml
# Binary classification with CrossEntropyLoss (wrong!)
system:
  criterion:
    _target_: torch.nn.CrossEntropyLoss  # For multi-class!
```

**Fix:**
```yaml
# Use BCEWithLogitsLoss for binary
system:
  criterion:
    _target_: torch.nn.BCEWithLogitsLoss  # For binary classification
```

### Example 3: Adapter Argument Mismatch

**Error:**
```
TypeError: __call__() got an unexpected keyword argument 'preds'
```

**User's config:**
```yaml
system:
  metrics:
    train:
      - _target_: torchmetrics.Accuracy
        # Expects 'preds' and 'target' kwargs
  adapters:
    train:
      metrics:
        _target_: lighter.adapters.MetricsAdapter
        # Using positional (default) instead of kwargs
```

**Fix:**
```yaml
system:
  adapters:
    train:
      metrics:
        _target_: lighter.adapters.MetricsAdapter
        pred_argument: "preds"    # Match metric signature
        target_argument: "target"
```

## Getting Help

When asking for help, include:

1. **Your config file** (or relevant section)
2. **Full error message** (including traceback)
3. **Lighter version:** `lighter --version`
4. **Python/PyTorch versions:** `python --version`, `python -c "import torch; print(torch.__version__)"`
5. **What you've tried** so far

### Resources

1. **Documentation:** Search this site
2. **FAQ:** [Common questions](../faq.md)
3. **Examples:** Check `projects/` directory in the repo
4. **PyTorch Lightning:** [Lightning docs](https://lightning.ai/docs/pytorch/stable/) for Trainer issues
5. **Discord:** [Join community](https://discord.gg/zJcnp6KrUp)
6. **GitHub Issues:** [Report bugs](https://github.com/project-lighter/lighter/issues)

## Summary

Most issues fall into these categories:

| Category | Quick Fix |
|----------|-----------|
| Config syntax | Check YAML indentation, quotes, colons |
| References | Use `::` for config, `.` for Python |
| Memory | Reduce batch size, use mixed precision |
| Speed | Increase num_workers, enable pin_memory |
| DDP | Enable sync_batchnorm, use rank_zero_only |
| Adapters | Add print transforms to inspect shapes |
| Metrics | Check mode has metrics, verify adapter args |

**Pro tip:** Most errors can be caught early with `trainer::fast_dev_run=true`!
