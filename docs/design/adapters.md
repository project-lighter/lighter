# The Adapter Pattern

## The Problem

Different ML components may expect different data formats. Consider a scenario where:

- Dataset returns dictionaries of tensors
- Model expects tensors
- Loss function needs specific argument order
- Metrics need different format than loss

Traditionally, you'd implement a pipeline specific to this scenario. This tightly couples components, making reuse and experimentation difficult.

## The Solution: Adapters

![System Data Flow](../assets/images/overview_system.png)
*Data flow through Lighter's System. Adapters bridge components with incompatible interfaces.*

In software engineering, the [adapter pattern](https://refactoring.guru/design-patterns/adapter) allows incompatible interfaces to work together. Lighter uses adapters to handle variability in data formats.

## Lighter's Adapter Types

| Adapter | Purpose | When to Use |
|---------|---------|-------------|
| **BatchAdapter** | Extract data from batches | Different dataset formats |
| **CriterionAdapter** | Format loss inputs | Custom loss functions |
| **MetricsAdapter** | Format metric inputs | Third-party metrics |
| **LoggingAdapter** | Transform before logging | Visualization needs |

## Example: Task-Agnostic Configuration

```yaml
adapters:
  train:
    criterion:
      _target_: lighter.adapters.CriterionAdapter
      pred_transforms:   # Apply sigmoid before loss
        _target_: torch.sigmoid
      pred_argument: 0   # Map pred to first argument
      target_argument: 1 # Map target to second argument
```

This enables **any task**—classification, segmentation, self-supervised learning—without framework modifications.

## Under the Hood

Adapters are invoked in this order during training:

1. **BatchAdapter** - Extract input/target from batch
2. **Forward pass** - Model processes input
3. **CriterionAdapter** - Format for loss computation
4. **MetricsAdapter** - Format for metric computation
5. **LoggingAdapter** - Transform for visualization

## Practical Usage

For detailed adapter configuration and examples, see:

- [Adapters How-To Guide](../how-to/adapters.md) - Complete usage guide
- [Metrics Guide](../how-to/metrics.md) - Using MetricsAdapter
- [Writers Guide](../how-to/writers.md) - Using LoggingAdapter
