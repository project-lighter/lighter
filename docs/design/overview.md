---
title: Architecture & Design
---

# Architecture & Design Overview

Lighter is a configuration-driven deep learning framework that separates experimental setup from code implementation.

## Core Architecture

![Lighter Overview](../assets/images/overview_all.png)
*Figure: Lighter's three-component (bolded) architecture. Config parses YAML definitions, System encapsulates DL components, and Trainer executes training.*

### 1. Config
Transforms YAML experiment definitions into Python objects using Sparkwheel. One config file = one reproducible experiment.

[→ Configuration guide](../how-to/configuration.md)

### 2. System
Orchestrates your deep learning pipeline—model, optimizer, loss, metrics, data. Extends PyTorch Lightning's LightningModule.

[→ System internals](system.md)

### 3. Trainer
PyTorch Lightning's Trainer executes experiments with multi-GPU, mixed precision, gradient accumulation, and checkpointing.

[→ Running experiments](../how-to/run.md)

## Stages and Modes

A key concept in Lighter is the distinction between **Stages** and **Modes**:

### Stages
Stages are what you invoke from the CLI. They represent high-level operations:

- `lighter fit` - Train and validate a model
- `lighter validate` - Run validation only
- `lighter test` - Evaluate on test set
- `lighter predict` - Generate predictions

### Modes
Modes are internal execution contexts that the System uses during a stage. Each mode has its own dataloader, metrics, and adapters:

- `train` - Training loop with backpropagation
- `val` - Validation loop (no gradients)
- `test` - Testing loop (no gradients)
- `predict` - Inference loop (no targets or loss)

### Stage-to-Mode Mapping

Each stage executes one or more modes:

```
FIT       → [train, val]  # Training with validation
VALIDATE  → [val]         # Validation only
TEST      → [test]        # Testing only
PREDICT   → [predict]     # Inference only
```

This means when you run `lighter fit`, the system will execute both training and validation modes. When you run `lighter validate`, only the validation mode executes.

**Code reference**: `src/lighter/engine/runner.py:26-31`

## Auto Config Pruning

Lighter automatically prunes (removes) unused components from your configuration based on the stage you're running. This allows you to define a single comprehensive configuration file that works for all stages.

### What Gets Pruned

The Runner (`src/lighter/engine/runner.py:80-112`) automatically removes:

1. **Unused mode components**: Dataloaders and metrics for modes not required by the current stage
   - Example: When running `lighter test`, train and val dataloaders/metrics are removed

2. **Training-only components** (for non-FIT stages):
   - Optimizer
   - Learning rate scheduler
   - Criterion (except for VALIDATE stage, which needs it for computing validation loss)

3. **Stage-specific arguments**: Arguments defined for other stages are removed
   - Example: `args::fit` is removed when running `lighter test`

### Why This Matters

Configuration pruning means you can:

- **Define once, use everywhere**: Write one config with train/val/test dataloaders, then use it for any stage
- **Avoid duplication**: No need for separate train.yaml, test.yaml, predict.yaml files
- **Reduce errors**: The system ensures only relevant components are used for each stage

### Example

```yaml
system:
  dataloaders:
    train: # Used only in FIT stage
      _target_: torch.utils.data.DataLoader
      # ... config ...

    val: # Used in FIT and VALIDATE stages
      _target_: torch.utils.data.DataLoader
      # ... config ...

    test: # Used only in TEST stage
      _target_: torch.utils.data.DataLoader
      # ... config ...

    predict: # Used only in PREDICT stage
      _target_: torch.utils.data.DataLoader
      # ... config ...

  optimizer: # Pruned for VALIDATE, TEST, PREDICT stages
    _target_: torch.optim.Adam
    lr: 0.001

  criterion: # Pruned for TEST, PREDICT stages
    _target_: torch.nn.CrossEntropyLoss
```

When you run `lighter test`, the Runner automatically removes the train, val, and predict dataloaders, as well as the optimizer and criterion, keeping only what's needed for testing.

## System Data Flow

Understanding how data flows through the System is crucial for working with Lighter effectively. The `System._step()` method (`src/lighter/system.py:74-94`) orchestrates this flow:

### The Pipeline

```
1. Batch (from DataLoader)
   ↓
2. BatchAdapter → [input, target, identifier]
   ↓
3. Model.forward(input) → prediction
   ↓ (or Inferer in val/test/predict modes)
   ↓
4. CriterionAdapter → Criterion(pred, target) → loss
   ↓
5. MetricsAdapter → Metrics(pred, target) → metric values
   ↓
6. LoggingAdapter → Logger
   ↓
7. Output dict returned to callbacks
```

### Step-by-Step Breakdown

**1. Batch Preparation** (`System._prepare_batch`)
- The raw batch from the DataLoader is passed to the BatchAdapter
- Returns: `(input, target, identifier)` tuple
- Identifier is optional and used for tracking samples

**2. Forward Pass** (`System.forward`)
- Input goes through `model.forward()` to produce predictions
- In val/test/predict modes, the Inferer replaces model.forward() if specified
- Automatically injects `epoch` and `step` arguments if model accepts them

**3. Loss Calculation** (`System._calculate_loss`)
- Only in train and val modes (test/predict skip this)
- CriterionAdapter transforms data before passing to criterion
- Supports dict-based losses with sublosses (must include 'total' key)

**4. Metrics Calculation** (`System._calculate_metrics`)
- Only in train, val, and test modes (predict skips this)
- MetricsAdapter transforms data before passing to metrics
- Returns None if no metrics are defined

**5. Logging** (`System._log_stats`)
- Logs loss and metrics to the logger
- Automatically logs optimizer stats (lr, momentum, beta) once per epoch in train mode

**6. Output Preparation** (`System._prepare_output`)
- LoggingAdapter can transform data for cleaner callback access
- Returns a dictionary with all step information

### The Output Dictionary

Each step returns a dictionary with the following keys:

```python
{
    "identifier": batch_identifier,  # Optional, for tracking
    "input": input_data,
    "target": target_data,
    "pred": predictions,
    "loss": loss_value,              # None in test/predict
    "metrics": metrics_dict,         # None in predict
    "step": global_step,
    "epoch": current_epoch,
}
```

This dictionary is accessible in callbacks, allowing you to write predictions to disk, visualize results, or perform custom analysis.

**Code references**:
- Data flow orchestration: `src/lighter/system.py:74-94`
- Dict-based loss handling: `src/lighter/system.py:156-160`
- Epoch/step injection: `src/lighter/system.py:121-126`

### Key Behaviors

- **Inferer replaces forward()**: In val/test/predict modes, if an inferer is specified, it's called instead of `model.forward()`. This is useful for inference-specific logic like sliding window or test-time augmentation.

- **Dict-based losses**: If your criterion returns a dictionary of sublosses, it must include a `'total'` key that combines all sublosses. This is used for backpropagation.

- **Mode-specific adapters**: Each mode (train/val/test/predict) has its own set of adapters, allowing different preprocessing for different stages.

- **Automatic optimizer stats**: Learning rate, momentum, and beta values are logged automatically once per epoch during training.

## The Adapter Pattern

Adapters make Lighter task-agnostic by handling data format differences between components.

[→ Learn more about adapters](adapters.md)

## Design Philosophy

Lighter follows four core principles: **Configuration over Code**, **Composition over Inheritance**, **Convention over Configuration**, and **Separation of Concerns**.

[→ Understand the philosophy](philosophy.md)

## Framework Comparison

Lighter's goal is to brings reproducibility and structure, while keeping you in full control of your code. This is different from other configuration-driven frameworks that provide higher-level abstractions.

| Feature | **Lighter** | **[Ludwig](https://github.com/ludwig-ai/ludwig)** | **[Quadra](https://github.com/orobix/quadra)** | **[GaNDLF](https://github.com/mlcommons/GaNDLF)** |
|---|---|---|---|---|
| **Primary Focus** | Config-driven, task-agnostic DL | Config-driven, multi-task DL | Config-driven computer vision | Config-driven medical imaging |
| **Configuration** | YAML (Sparkwheel) | YAML (Custom) | YAML (Hydra) | YAML (Custom) |
| **Abstraction** | Medium. Extends PyTorch Lightning, expects standard PyTorch components. | High. Provides pre-built flows for various tasks. | High. Pre-defined structures for computer vision. | High. Pre-defined structures for medical imaging. |
| **Flexibility** | High. New components are added via project module. | Medium. Adding new components requires code editing. | Low. Adding new components requires code editing. | Low. Adding new components requires code editing. |
| **Use Case** | Organized experimentation | Production-level applications | Traditional computer vision | Established medical imaging methods |

Lighter is the tool for you if you like PyTorch's flexibility but want to manage your experiments in a structured and reproducible way.

## Next Steps

- Deep dive into [the Adapter Pattern](adapters.md)
- Understand [Design Philosophy](philosophy.md)
- Get started with the [Zero to Hero tutorial](../tutorials/get-started.md)
