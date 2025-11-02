# Custom Metrics: Beyond Standard Evaluation

Metrics are the compass that guides your model development. While `torchmetrics` provides excellent built-in metrics, real-world projects often need custom evaluation logic. This guide shows you how to create powerful custom metrics that provide deep insights into your model's behavior.

## Quick Start: Your First Custom Metric in 30 Seconds 🚀

```python
# my_project/metrics/weighted_accuracy.py
from torchmetrics import Metric
import torch

class WeightedAccuracy(Metric):
    """Accuracy that cares more about certain classes."""
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights
        self.add_state("weighted_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_weight", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        pred_classes = preds.argmax(dim=1)
        correct = pred_classes == target
        weights = torch.tensor([self.class_weights[t.item()] for t in target])
        self.weighted_correct += (correct * weights).sum()
        self.total_weight += weights.sum()

    def compute(self):
        return self.weighted_correct / self.total_weight
```

Use it in your config:
```yaml
system:
    metrics:
        val:
            - _target_: my_project.metrics.WeightedAccuracy
              class_weights: [1.0, 2.0, 5.0]  # Class 2 is 5x more important
```

## Core Concepts: The Metric Trinity 🏆

Every custom metric needs three essential components:

| Component | Purpose | When Called |
|-----------|---------|-------------|
| **1. `add_state()`** | Register variables to track | Once at initialization |
| **2. `update()`** | Process batch & accumulate stats | Every batch |
| **3. `compute()`** | Calculate final metric value | End of epoch/validation |

**The lifecycle flow:**
1. **Initialize** → Set up state variables
2. **Update** (repeated) → Process each batch
3. **Compute** → Calculate final metric

## Creating a Custom Metric: Step-by-Step

Let's walk through the steps of creating a custom metric in Lighter using `torchmetrics`. We'll create a simple example custom metric called `MyCustomMetric` for binary classification, which calculates a variation of accuracy.

1.  **Subclass `torchmetrics.Metric`**:
    First, create a new Python file (e.g., `my_project/metrics/my_custom_metric.py`) within your project to define your custom metric class. Start by importing `torchmetrics.Metric` and subclassing it.

    ```python title="my_project/metrics/my_custom_metric.py"
    from torchmetrics import Metric
    import torch

    class MyCustomMetric(Metric):
        def __init__(self):
            super().__init__()
            # ... (state initialization will be added in the next step) ...

        def update(self, preds: torch.Tensor, target: torch.Tensor):
            # ... (update logic will be added in the next step) ...
            pass

        def compute(self):
            # ... (compute logic will be added in the next step) ...
            pass
    ```

2.  **Initialize Metric State with `add_state()`**:
    In the `__init__` method, use `self.add_state()` to initialize state variables for accumulated statistics. For `MyCustomMetric`, track correct and total predictions:

    ```python title="my_project/metrics/my_custom_metric.py"
    from torchmetrics import Metric
    import torch

    class MyCustomMetric(Metric):
        def __init__(self):
            super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum") # Tracks correct predictions
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")   # Tracks total predictions
    ```

    - Registers "correct" state variable:
        - Initializes to a PyTorch tensor of 0.
        - `dist_reduce_fx="sum"`: Reduces state across distributed processes by summing.
    - Registers "total" state variable:
        - Initializes to a PyTorch tensor of 0.
        - `dist_reduce_fx="sum"`: Reduces state across distributed processes similarly.

3.  **Implement `update()` Method**:
    The `update()` method processes each batch of predictions and targets. For `MyCustomMetric`, implement the following:

    1.  Convert probability predictions to binary (0/1).
    2.  Count correct predictions and update state variables.

    ```python title="my_project/metrics/my_custom_metric.py"
    from torchmetrics import Metric
    import torch

    class MyCustomMetric(Metric):
        def __init__(self):
            super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # 1. Convert probabilities to binary predictions
        # Convert probabilities to binary (0/1).
        #   - `binary_preds = (preds >= 0.5).int()`: Converts probabilities to binary predictions (0 or 1). # commented out to avoid repetition

        # 2. Count correct predictions and update state variables.
        # Count correct predictions and update state variables.
        #   - `self.correct += torch.sum(binary_preds == target)`: Increments `correct` state with batch's correct predictions. # commented out to avoid repetition
        #   - `self.total += target.numel()`: Increments `total` state with batch size. # commented out to avoid repetition
        self.correct += torch.sum(binary_preds == target)
        self.total += target.numel()
    ```

    - 1. Convert probability predictions to binary (0/1).
    - 2. Count correct predictions and update state variables.

4.  **Implement `compute()` Method**:
    The `compute()` method calculates the final metric value at the epoch end. For `MyCustomMetric`, calculate custom accuracy:

    ```python title="my_project/metrics/my_custom_metric.py"
    from torchmetrics import Metric
    import torch

    class MyCustomMetric(Metric):
        def __init__(self):
            super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        binary_preds = (preds >= 0.5).int()
        self.correct += torch.sum(binary_preds == target)
        self.total += target.numel()

    def compute(self):
        # Returns custom accuracy: correct predictions / total predictions
        return self.correct.float() / self.total
```

    - Returns custom accuracy: correct predictions / total predictions

5.  **Integrate with Lighter Configuration**:
    Reference your custom metric in `config.yaml` to use it during train/val/test.

    **Example `config.yaml`**:

    ```yaml title="config.yaml"
    project: my_project/ # Project root directory

    system:
      metrics:
        train:
          - _target_: torchmetrics.Accuracy
          - _target_: my_project.metrics.MyCustomMetric # Use custom metric

        val:
          - _target_: torchmetrics.Accuracy
          - _target_: my_project.metrics.MyCustomMetric
    ```

    This config uses both built-in `Accuracy` and `MyCustomMetric` during train/val stages.

## Practical Example: Domain-Specific Metric

```python
from torchmetrics import Metric
import torch

class DiceScore(Metric):
    """Dice coefficient for segmentation tasks."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Flatten predictions and targets
        preds = preds.view(-1)
        target = target.view(-1)

        # Calculate intersection and union
        intersection = (preds * target).sum()
        union = preds.sum() + target.sum()

        self.intersection += intersection
        self.union += union

    def compute(self):
        # Calculate Dice score
        dice = (2 * self.intersection + self.smooth) / (self.union + self.smooth)
        return dice
```

Use in config:
```yaml
system:
    metrics:
        val:
            - _target_: my_project.metrics.DiceScore
              smooth: 1e-6
```

## Key Optimization Tips ⚡

| Tip | Do | Don't |
|-----|----|---------|
| **Use Vectorization** | `(preds == target).sum()` | Loop through elements |
| **Accumulate Stats** | Store sums and counts | Store all predictions |
| **Handle Edge Cases** | Check for zero division | Assume valid inputs |

## Common Pitfalls 🛡️

| Pitfall | Solution |
|---------|----------|
| **State accumulation across epochs** | Lighter resets automatically |
| **Wrong distributed reduction** | Use `dist_reduce_fx="sum"` for counts |
| **Type mismatches** | Convert tensors to same dtype |

## Quick Reference Card 📄

```python
# Minimal custom metric template
from torchmetrics import Metric
import torch

class YourMetric(Metric):
    def __init__(self, your_param=1.0):
        super().__init__()
        # 1. Register state variables
        self.add_state("state_var", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # 2. Process batch and update state
        self.state_var += your_computation(preds, target)

    def compute(self):
        # 3. Calculate final metric
        return self.state_var / normalization_factor
```

## Recap and Next Steps

You now have the power to create sophisticated custom metrics:

🎨 **What You Learned:**

- Core metric lifecycle: `add_state()` → `update()` → `compute()`
- Performance optimization techniques
- Testing strategies for robust metrics
- Common patterns for classification, regression, and calibration

💡 **Pro Tip:** Start simple, test thoroughly, optimize later!

## Related Guides
- [Flows](flows.md) - Transform data for metrics
- [Writers](writers.md) - Save metric results
