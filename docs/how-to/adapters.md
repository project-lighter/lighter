# Adapters: Data Flow Control

Adapters are the **secret sauce** that makes Lighter incredibly flexible. They act as intelligent translators between different components of your pipeline, ensuring data flows correctly regardless of format differences.

## The Data Flow Problem

In a typical training step, data flows sequentially: a **batch** from the dataloader is fed to the **model** to produce a **prediction**. This prediction, along with the **target** from the batch, is then used by the **loss function**, **metrics**, and **logger**.

The problem is that each component might have different expectations for the data it receives:

- A `Dataset` might return a dictionary of tensors, but the `model` only needs a specific tensor.
- A `loss function` might expect `(prediction, target)`, while another expects `(target, prediction)`.
- A `metric` might need class indices, but the model outputs probabilities.

**Without adapters**, you would have to write glue code inside your model or training loop to handle these mismatches. This makes your components less reusable.

**With Lighter's adapters**, you define these translations in your configuration, keeping your components clean and independent.

## The Solution: Adapters

Lighter provides four types of adapters, each designed to intercept and transform data at a specific point in the data flow:

| Adapter | Purpose | When to Use |
|---|---|---|
| **BatchAdapter** | Extracts and formats data from the dataloader's batch. | When your dataset has a different structure than the expected `(input, target)`. |
| **CriterionAdapter** | Prepares `input`, `target`, and `pred` for the loss function. | When your loss function has specific argument or data format requirements. |
| **MetricsAdapter** | Prepares `input`, `target`, and `pred` for metrics. | When your metrics have specific argument or data format requirements. |
| **LoggingAdapter** | Prepares `input`, `target`, and `pred` for logging. | When you need to transform data for visualization. |

Here are some common challenges and how adapters solve them. Don't worry about the details yet—we'll dive deeper into each adapter below.:

| Component / Scenario | Common Challenge | Adapter Solution |
| :--- | :--- | :--- |
| **Batch** 📦 | Dataset returns a `dict` (e.g., `{"x": ..., "y": ...}`), but the pipeline needs `input` and `target`. | Use `BatchAdapter` to map dictionary keys to `input` and `target` (e.g., `input_accessor="x"`). |
| **Batch in Self-Supervision** | The training step has no `target`. | Use `BatchAdapter` and set `target_accessor` to `None`. |
| **Loss Function** 📉 | Loss function expects `loss(target, pred)`, but Lighter's default is `loss(pred, target)`. | Use `CriterionAdapter` to reorder arguments (e.g., `pred_argument=1`, `target_argument=0`). |
| **Metrics** 📏 | A metric needs class indices, but the model outputs probabilities. | Use `MetricsAdapter` with `pred_transforms` to apply `torch.argmax` before the metric calculation. |
| **Logging** 📊 | Logger expects RGB images, but the data is grayscale. | Use `LoggingAdapter` with `input_transforms` to repeat the channel dimension. |

## Execution Order

Understanding when each adapter is called is crucial for debugging and designing your pipeline. Here's the complete data flow during a training step:

```
Step 1: DataLoader produces batch
        ↓
Step 2: BatchAdapter
        └─→ Extracts: (input, target, identifier)
        ↓
Step 3: Model.forward(input) or Inferer
        └─→ Produces: prediction
        ↓
Step 4: CriterionAdapter (train/val modes only)
        ├─→ Transforms: input, target, pred
        └─→ Maps to loss function arguments
        ↓
Step 5: Loss function computes loss
        ↓
Step 6: MetricsAdapter (train/val/test modes)
        ├─→ Transforms: input, target, pred
        └─→ Maps to metrics arguments
        ↓
Step 7: Metrics compute values
        ↓
Step 8: LoggingAdapter
        ├─→ Transforms: input, target, pred
        └─→ Prepares for logger/callbacks
        ↓
Step 9: Logger and callbacks receive data
```

### Key Points

- **BatchAdapter** runs first, before the model sees any data
- **CriterionAdapter** runs only in train and val modes (skipped in test/predict)
- **MetricsAdapter** runs only in train, val, and test modes (skipped in predict)
- **LoggingAdapter** runs last, after all computations are done
- Each mode (train/val/test/predict) can have its own set of adapters

### Practical Implications

1. **BatchAdapter transforms are executed on every batch**: Keep them lightweight, or move expensive operations to your Dataset's `__getitem__`.

2. **CriterionAdapter and MetricsAdapter run after model forward**: You can safely apply post-processing like `argmax` or `sigmoid` here without affecting the model.

3. **LoggingAdapter is for visualization only**: Transforms here don't affect training—use them for detaching tensors, converting to CPU, or formatting for display.

4. **Mode-specific adapters**: You can configure different adapters for train vs val vs test:
   ```yaml
   system:
     adapters:
       train:
         batch: ...
         criterion: ...
       val:
         batch: ...  # Can be different from train
         criterion: ...
   ```

For a deeper understanding of the complete System data flow, see [System Internals](../design/system.md#system-data-flow).

## Deep Dive

### BatchAdapter

The `BatchAdapter` is responsible for extracting `input`, `target`, and an optional `identifier` from each batch produced by your `DataLoader`. This is the first and most common adapter you'll use.

Lighter needs to know how to get the following from a batch:

| Item | Description |
|-----------|-------------|
| Input | The data fed into the model (e.g., images, text). |
| Target **(optional)** | The ground truth data (e.g., labels, masks) used for loss calculation. |
| Identifier **(optional)** | A unique ID for the data point (e.g., filename, patient ID). |

!!! note
    By default, Lighter assumes the batch is an `(input, target)` tuple:
    `BatchAdapter(input_accessor=0, target_accessor=1)`

You can specify `accessors` to handle different batch structures:

| Accessor Type | Description | Example |
| --- | --- | --- |
| **Integer Index** | Access elements by position in a list or tuple batch. | `input_accessor: 0` |
| **String Key** | Access elements by key in a dictionary batch. | `input_accessor: "image"` |
| **Callable** | Use a function for complex logic. | `target_accessor: $lambda batch: one_hot(batch[1])` |

#### Example 1: Dictionary-based Dataset

If your dataset returns a dictionary, you can map the keys to `input`, `target`, and `identifier`.

```yaml
# Problem: Dataset returns a dict, but the model needs tensors.
system:
  adapters:
    train:
      batch:
        _target_: lighter.adapters.BatchAdapter
        input_accessor: "image"
        target_accessor: "mask"
        identifier_accessor: "patient_id"
```

#### Example 2: Self-Supervised Learning

In self-supervised learning, you might not have a `target`. You can set its accessor to `None`.

```yaml
# Problem: No targets in this training phase.
system:
  adapters:
    train:
      batch:
        _target_: lighter.adapters.BatchAdapter
        input_accessor: 0
        target_accessor: null  # No targets!
```

For more details, see the [`BatchAdapter` documentation](../../reference/adapters/#lighter.adapters.BatchAdapter).

### CriterionAdapter

The `CriterionAdapter` acts as a bridge between your model's prediction and your loss function. It allows you to:

1.  **Map** `pred`, `target`, and `input` to the arguments of your loss function.
2.  **Transform** these tensors before they are passed to the loss function.

**Argument Mappers** can be:

| Mapper Type | Description | Example |
| --- | --- | --- |
| **Integer Index** | Map to a positional argument. | `pred_argument: 1`, `target_argument: 0` |
| **String Key** | Map to a keyword argument. | `pred_argument: "prediction"` |
| **`None`** | Don't pass this tensor to the loss function. | `input_argument: None` |

**Transforms** are functions applied to the tensors before mapping.

| Transform Type | Description | Example |
| --- | --- | --- |
| **Callable** | A function or list of functions to apply. | `pred_transforms: [_target_: torch.sigmoid]` |

#### Example: Custom Argument Order and Activation

If your loss function expects `(target, pred)` and requires sigmoid on the predictions:

```yaml
# Problem: Loss function has a non-standard signature and needs activated predictions.
system:
  adapters:
    train:
      criterion:
        _target_: lighter.adapters.CriterionAdapter
        pred_argument: 1  # Pass 'pred' as the 2nd argument
        target_argument: 0 # Pass 'target' as the 1st argument
        pred_transforms:
          - _target_: torch.sigmoid
```

For more details, see the [`CriterionAdapter` documentation](../../reference/adapters/#lighter.adapters.CriterionAdapter).

### MetricsAdapter

The `MetricsAdapter` is identical in configuration to the `CriterionAdapter`, but for your metrics. You can use it to map and transform `pred`, `target`, and `input` before they are fed into your `torchmetrics` functions.

#### Example: Preparing Predictions for a Metric

If a metric requires class indices from your model's output probabilities:

```yaml
# Problem: Metric expects class indices, not probabilities.
system:
  adapters:
    val:
      metrics:
        _target_: lighter.adapters.MetricsAdapter
        pred_argument: "preds"
        target_argument: "target"
        pred_transforms:
          - _target_: torch.argmax
            dim: 1
```

For more details, see the [`MetricsAdapter` documentation](../../reference/adapters/#lighter.adapters.MetricsAdapter).

### LoggingAdapter

The `LoggingAdapter` is used to transform `input`, `target`, and `pred` tensors just before they are sent to the logger (e.g., for image visualization in TensorBoard). It only supports transforms, not argument mapping.

#### Example: Visualizing Grayscale Images

If you want to log a single-channel image and your logger expects a 3-channel RGB image:

```yaml
# Problem: Logger expects a 3-channel image, but data is grayscale.
system:
  adapters:
    train:
      logging:
        _target_: lighter.adapters.LoggingAdapter
        input_transforms: "$lambda x: x.repeat(1, 3, 1, 1)" # Convert to 3-channel
```

For more details, see the [`LoggingAdapter` documentation](../../reference/adapters/#lighter.adapters.LoggingAdapter).

## Advanced Usage

### Custom Adapters

For highly complex scenarios, you can implement your own adapter by inheriting from Lighter's base adapters.

```python
# my_project/adapters.py
from lighter.adapters import BatchAdapter

class MultiModalBatchAdapter(BatchAdapter):
    """A custom adapter for multi-modal data."""
    def __call__(self, batch):
        # Custom logic to unpack a complex batch
        return {
            "image": batch["image_data"],
            "text": batch["text_embeddings"],
            "tabular": batch["clinical_features"]
        }, batch["diagnosis"], batch["id"]
```

Then, use it in your config:

```yaml
system:
  adapters:
    train:
      batch:
        _target_: my_project.adapters.MultiModalBatchAdapter
```

## Tips and Troubleshooting

| Tip / Issue | Solution |
|---|---|
| **Performance** | Put expensive transforms in your `Dataset`, not in adapters, to leverage multi-worker data loading. |
| **Debugging** | Add a print transform (`$lambda x: print(x.shape) or x`) to inspect tensor shapes at any point in the flow. |
| **`KeyError` in batch** | Your `input_accessor` or `target_accessor` might be wrong. Use a print transform to inspect the batch keys/indices. |
| **Wrong argument order** | Double-check the signature of your loss/metric function and use named arguments in `CriterionAdapter` or `MetricsAdapter` for clarity. |
| **Reusing configs** | Use raw references (`%`) to reuse adapters: `val: "%system::adapters::train::batch"` creates a new instance with the same config. |

## Complete Example: Segmentation Pipeline

Here’s how adapters work together in a complete segmentation pipeline:


```yaml
system:
  adapters:
    train:
      # 1. Extract 'image' and 'mask' from the batch dictionary.
      batch:
        _target_: lighter.adapters.BatchAdapter
        input_accessor: "image"
        target_accessor: "mask"
        identifier_accessor: "patient_id"

      # 2. Pass prediction and target to the loss function and apply softmax.
      criterion:
        _target_: lighter.adapters.CriterionAdapter
        pred_argument: 0
        target_argument: 1
        pred_transforms:
          - _target_: torch.nn.functional.softmax
            dim: 1

      # 3. Pass prediction and target to metrics by name and apply argmax.
      metrics:
        _target_: lighter.adapters.MetricsAdapter
        pred_argument: "preds"
        target_argument: "target"
        pred_transforms:
          - _target_: torch.argmax
            dim: 1

    # 4. Reuse the same batch adapter for validation.
    val:
      batch: "%system::adapters::train::batch"
      # Criterion and Metrics adapters for 'val' would also be defined here.
```

## Recap and Next Steps

Adapters are what make Lighter truly flexible:

✅ **Key Benefits:**

- Handle any data format without code changes
- Connect incompatible components seamlessly
- Transform data at the right pipeline stage
- Debug and monitor data flow easily

🎯 **Best Practices:**

- Keep transforms simple and composable
- Move expensive operations to datasets
- Use debug prints during development
- Reuse configurations with YAML anchors

## Related Guides
- [Metrics](metrics.md) - Using MetricsAdapter
- [Writers](writers.md) - Using LoggingAdapter
- [Inferers](inferers.md) - Inference-time adaptation
