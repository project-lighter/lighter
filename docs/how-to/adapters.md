# Adapters: Data Flow Control

Adapters are the **secret sauce** that makes Lighter incredibly flexible. They act as intelligent translators between different components of your pipeline, ensuring data flows correctly regardless of format differences.

## Why Adapters Matter 🎟️

Imagine connecting ML components that expect different data formats:

**Without adapters:** You'd need to modify code for each component combination

**With adapters:** You configure the connections once, and Lighter handles the format conversion

| Component | What it expects | Adapter handles |
|-----------|----------------|------------------|
| Dataset 📦 | Returns dictionaries | Extract tensors |
| Model 🧠 | Needs tensors | Format inputs |
| Loss 📉 | Specific argument order | Reorder arguments |
| Metrics 📏 | Named arguments | Map predictions |
| Logger 📊 | RGB images | Convert grayscale |

## Real-World Scenarios

### Scenario 1: Medical Imaging Pipeline
```yaml
# Problem: Dataset returns dict, but model expects tensor
system:
    adapters:
        train:
            batch:
                _target_: lighter.adapters.BatchAdapter
                input_accessor: "image"  # Extract from dict
                target_accessor: "mask"  # Extract mask
                identifier_accessor: "patient_id"  # Track patient
```

### Scenario 2: Multi-Task Learning
```yaml
# Problem: Multiple losses need different arguments
system:
    adapters:
        train:
            criterion:
                _target_: lighter.adapters.CriterionAdapter
                pred_argument: "predictions"  # Named argument
                target_argument: "labels"     # Named argument
                input_argument: "features"    # Pass input too
```

### Scenario 3: Self-Supervised Learning
```yaml
# Problem: No targets in SSL, only inputs
system:
    adapters:
        train:
            batch:
                _target_: lighter.adapters.BatchAdapter
                input_accessor: 0
                target_accessor: null  # No targets!
```

## Types of Adapters in Lighter

Lighter provides four adapter types, each designed to address specific customization needs:

| Adapter | Purpose | When to Use |
|---------|---------|-------------|
| **BatchAdapter** | Extract data from batches | Different dataset formats |
| **CriterionAdapter** | Format loss function inputs | Custom loss functions |
| **MetricsAdapter** | Format metric inputs | Third-party metrics |
| **LoggingAdapter** | Transform before logging | Visualization needs |

###  BatchAdapter

Lighter expects the following data structure:

| Item | Description |
|-----------|-------------|
| Input | The input data (e.g., images, text, audio) that is fed into the model. |
| Target **(optional)** | The target data (e.g., labels, segmentation masks) used to compute the loss. |
| Identifier **(optional)** | An identifier for the data (e.g., image filenames, patient IDs). |


Different datasets and tasks may use different batch structures. To address that, we provide [`lighter.adapters.BatchAdapter`](../../reference/adapters/#lighter.adapters.BatchAdapter) to customize how data is extracted from the batch. You can specify accessors to extract the `input`, `target`, and `identifier` tensors from the batch, regardless of the original batch structure.

!!! note
    By default, Lighter assumes that the batch is an `(input, target)` tuple, without identifier, defined by:
    ```
    BatchAdapter(input_accessor=0, target_accessor=1, identifier_accessor=None)
    ```

BatchAdapter **accessors** can be:

| Accessor Type | Description | Example |
| --- | --- | --- |
| Integer Index | For list or tuple batches, use an integer index to access elements by position. | `input_accessor=0, target_accessor=1` |
| String Key | For dictionary batches, use a string key (e.g., `"image"`) to access elements by key. | `input_accessor="image", target_accessor="label", identifier_accessor="name"` |
| Callable Function (Advanced) | For more complex batch structures or transformations, provide a callable function that takes the batch as input and returns the desired `input`, `target`, or `identifier`. | `input_accessor=0, target_accessor=lambda batch: one_hot_encode(batch[1])` |

Below, you can see an example of how to configure the `BatchAdapter` in your config file:

```yaml
system:
# ...
    adapters:
        train:
            batch:
                _target_: lighter.adapters.BatchAdapter
                input_accessor: "image"
                target_accessor: "label"
                identifier_accessor: "id"
```

For more information, see the [BatchAdapter documentation](../../reference/adapters/#lighter.adapters.BatchAdapter).


### CriterionAdapter


CriterionAdapter **argument mappers** specify how the `pred`, `target`, and `input` tensors are passed as arguments to your criterion function.  You can configure argument mappers and transforms for each of `pred`, `target`, and `input`.

CriterionAdapter **argument mappers** can be:

| Mapper Type | Description | Example |
| --- | --- | --- |
| Integer Index | For criterion functions expecting positional arguments, use an integer index to specify the argument position (starting from 0). | `pred_argument=1, target_argument=0` |
| String Key | For criterion functions expecting keyword arguments, use a string key (e.g., `"prediction"`) to specify the argument name. | `pred_argument="prediction", target_argument="ground_truth"` |
| `None` | If the criterion function doesn't require a specific tensor (`pred`, `target`, or `input`), set the corresponding argument mapper to `None`. | `input_argument=None` |

**Transforms** allow you to apply functions to `pred`, `target`, and `input` tensors before they are passed to the criterion.

| Transform Type | Description | Example |
| --- | --- | --- |
| Callable Function | Provide a callable function (or a list of callable functions) that takes the tensor as input and returns the transformed tensor. Transforms are defined using the MONAI Bundle syntax. | `pred_transforms: [_target_: torch.sigmoid]` |

Below, you can see an example of how to configure the `CriterionAdapter` in your config file:

```yaml title="config.yaml"
system:
    adapters:
        train:
            criterion:
                _target_: lighter.adapters.CriterionAdapter
                # Map 'pred' to the 2nd positional argument (index 1)
                pred_argument: 1
                # Map 'target' to the 1st positional argument (index 0)
                target_argument: 0
                # Apply sigmoid activation to predictions
                pred_transforms:
                    - _target_: torch.sigmoid
```

For more information, see the [CriterionAdapter documentation](../../reference/adapters/#lighter.adapters.CriterionAdapter).
### MetricsAdapter

The configuration of `MetricsAdapter` is **identical** to `CriterionAdapter`. You use **argument mappers** and **transforms** in the same way.

Below, you can see an example of how to configure the `MetricsAdapter` in your config file:

```yaml title="config.yaml"
system:
    adapters:
        val:
            metrics:
                _target_: lighter.adapters.MetricsAdapter
                # Map 'pred' to keyword argument "prediction"
                pred_argument: "prediction"
                # Map 'target' to keyword argument "ground_truth"
                target_argument: "ground_truth"
                # Convert pred to class labels
                pred_transforms:
                    - _target_: torch.argmax
                      dim: 1
```


For more information, see the [MetricsAdapter documentation](../../reference/adapters/#lighter.adapters.MetricsAdapter).
### LoggingAdapter

LoggingAdapter configuration focuses on applying **transforms** to `pred`, `target`, and `input` tensors before they are logged.

**Transforms** for LoggingAdapter are configured using `pred_transforms`, `target_transforms`, and `input_transforms`.

| Transform Type | Description | Example |
| --- | --- | --- |
| Callable Function | Provide a callable function (or a list of callable functions) that takes the tensor as input and returns the transformed tensor. Transforms are defined using the MONAI Bundle syntax. | `input_transforms: [_target_: monai.transforms.ToNumpy]` |

Below, you can see an example of how to configure the `LoggingAdapter` in your config file:

```yaml title="config.yaml"
system:
    adapters:
    train:
        logging:
            _target_: lighter.adapters.LoggingAdapter # Use the built-in LoggingAdapter
            # Turn grayscale into rgb by repeating the channels
            input_transforms: "$lambda x: x.repeat(1, 3, 1, 1)"
            pred_transforms:
                - _target_: torch.argmax
                  dim: 1
```


For more information, see the [LoggingAdapter documentation](../../reference/adapters/#lighter.adapters.LoggingAdapter).

## Practical Example: Custom Adapter

```python
# my_project/adapters/custom_adapter.py
from lighter.adapters import BatchAdapter

class MultiModalBatchAdapter(BatchAdapter):
    """Handle multi-modal data (image + text + tabular)."""
    def __call__(self, batch):
        return {
            "image": batch["image_data"],
            "text": batch["text_embeddings"],
            "tabular": batch["clinical_features"],
            "target": batch["diagnosis"]
        }
```

Use in config:
```yaml
system:
    adapters:
        train:
            batch:
                _target_: my_project.adapters.MultiModalBatchAdapter
```

## Pro Tips 💡

1. **Debug outputs**: Add print transforms to see data flow
2. **Reuse configs**: Use YAML anchors (`&` and `*`) for DRY configs
3. **Performance**: Put expensive transforms in datasets, not adapters
4. **Type consistency**: Ensure tensors have matching dtypes

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **KeyError in batch** | Print batch keys: `"$lambda b: print(b.keys()) or b"` |
| **Wrong argument order** | Use named arguments in CriterionAdapter |
| **Transform not applied** | Ensure transforms are in a list |
| **Memory issues** | Process on CPU, then move to GPU |

## Complete Example: Segmentation Pipeline

```yaml
system:
    adapters:
        train:
            batch:
                _target_: lighter.adapters.BatchAdapter
                input_accessor: "image"  # Extract from dict
                target_accessor: "mask"
                identifier_accessor: "patient_id"

            criterion:
                _target_: lighter.adapters.CriterionAdapter
                pred_argument: 0  # First positional arg
                target_argument: 1  # Second positional arg
                pred_transforms:
                    - _target_: torch.nn.functional.softmax
                      dim: 1

            metrics:
                _target_: lighter.adapters.MetricsAdapter
                pred_argument: "preds"  # Named argument
                target_argument: "target"
                pred_transforms:
                    - _target_: torch.argmax
                      dim: 1

        val:
            batch: "%system#adapters#train#batch"  # Reuse train config
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
