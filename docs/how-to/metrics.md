Metrics are key for evaluating deep learning models. Lighter integrates `torchmetrics` for defining metrics in PyTorch. While `torchmetrics` offers many built-in metrics, custom metrics are often needed for specific research problems.

This guide walks through creating and using custom metrics in Lighter, enabling deeper insights into model behavior and performance evaluation tailored to your needs.

## `torchmetrics` Basics for Custom Metrics

Lighter uses `torchmetrics` as the metric system foundation. To create custom metrics, understand these `torchmetrics` concepts:

- **Metric Class**: Base class for custom metrics, inheriting from `torchmetrics.Metric`. Provides structure and methods.
- **`update()` Method**: Called for each data batch during train/val/test. Accumulates statistics based on predictions and targets.
- **`compute()` Method**: Called at epoch end (or val/test end). Calculates final metric value from accumulated stats.
- **`add_state()`**: Method to store accumulated stats in custom metric class. Registers state variables managed by `torchmetrics` (distributed computation, state resetting).

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

## Complete Custom Metric Example

Here's the complete code for our example custom metric, `MyCustomMetric` (in `my_project/metrics/my_custom_metric.py`):

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
        return self.correct.float() / self.total
```

And here's how you would use it in your `config.yaml`:

```yaml title="config.yaml"
project: my_project/

system:
  metrics:
    train:
      - _target_: torchmetrics.Accuracy
      - _target_: my_project.metrics.MyCustomMetric

    val:
      - _target_: torchmetrics.Accuracy
      - _target_: my_project.metrics.MyCustomMetric
```

## Recap and Next Steps

Creating custom metrics in Lighter using `torchmetrics` involves these key steps:

1.  **Subclass `torchmetrics.Metric`**.
2.  **Initialize State with `add_state()`** in `__init__`.
3.  **Implement `update()`** to process batches and update state.
4.  **Implement `compute()`** to calculate final metric value.
5.  **Integrate in `config.yaml`** using `_target_`.

This enables extending Lighter with custom metrics for tailored model evaluation.
