# The Flow

## The Problem

Different ML components may expect different data formats. Consider a scenario where:

- Dataset returns dictionaries of tensors
- Model expects tensors
- Loss function needs specific argument order
- Metrics need different format than loss

Traditionally, you'd implement a pipeline specific to this scenario. This tightly couples components, making reuse and experimentation difficult.

## The Solution: Flows

![System Data Flow](../assets/images/overview_system.png)
*Data flow through Lighter's System. Flows bridge components with incompatible interfaces.*

In Lighter, a Flow defines the entire step logic, from unpacking the batch to defining the output. It follows a "convention over configuration" philosophy. The output of the model is always stored in the context as 'pred', and the output of the criterion is always stored as 'loss'.

## Lighter's Flow

| Component | Purpose | When to Use |
|---|---|---|
| **batch** | Extract data from batches | Different dataset formats |
| **model** | Format model inputs | Custom model input |
| **criterion** | Format loss inputs | Custom loss functions |
| **metrics** | Format metric inputs | Third-party metrics |
| **output** | Define the step's output | Custom output |
| **logging** | Transform before logging | Visualization needs |

## Example: Task-Agnostic Configuration

```yaml
flows:
  train:
    _target_: lighter.Flow
    batch: ["input", "target"]
    model: ["input"]
    criterion: ["pred", "target"]
    metrics: ["pred", "target"]
    output:
      loss: "loss"
      pred: "pred"
```

This enables **any task**—classification, segmentation, self-supervised learning—without framework modifications.

## Under the Hood

Flows are invoked in this order during training:

1. **batch** - Extract input/target from batch
2. **model** - Model processes input
3. **criterion** - Format for loss computation
4. **metrics** - Format for metric computation
5. **logging** - Transform for visualization
6. **output** - Define the step's output

## Practical Usage

For detailed Flow configuration and examples, see:

- [Flows How-To Guide](../how-to/flows.md) - Complete usage guide
