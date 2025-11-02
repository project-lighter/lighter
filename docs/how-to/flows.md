# Flows: Data Flow Control

Flows are the **secret sauce** that makes Lighter incredibly flexible. They act as intelligent translators between different components of your pipeline, ensuring data flows correctly regardless of format differences.

## The Data Flow Problem

In a typical training step, data flows sequentially: a **batch** from the dataloader is fed to the **model** to produce a **prediction**. This prediction, along with the **target** from the batch, is then used by the **loss function**, **metrics**, and **logger**.

The problem is that each component might have different expectations for the data it receives:

- A `Dataset` might return a dictionary of tensors, but the `model` only needs a specific tensor.
- A `loss function` might expect `(prediction, target)`, while another expects `(target, prediction)`.
- A `metric` might need class indices, but the model outputs probabilities.

**Without Flows**, you would have to write glue code inside your model or training loop to handle these mismatches. This makes your components less reusable.

**With Lighter's Flows**, you define these translations in your configuration, keeping your components clean and independent.

## The Solution: Flows

Lighter provides a `Flow` class that allows you to define the data flow for each step (`train`, `val`, `test`, `predict`). A `Flow` is defined by a set of components, each responsible for a specific part of the data flow:

| Component | Purpose | When to Use |
|---|---|---|
| **batch** | Extracts and formats data from the dataloader's batch. | When your dataset has a different structure than the expected `(input, target)`. |
| **model** | Prepares `input` for the model. | When your model has specific argument or data format requirements. |
| **criterion** | Prepares `pred` and `target` for the loss function. | When your loss function has specific argument or data format requirements. |
| **metrics** | Prepares `pred` and `target` for metrics. | When your metrics have specific argument or data format requirements. |
| **output** | Defines the output of the step. | When you need to return specific values from the step. |
| **logging** | Prepares `input`, `target`, and `pred` for logging. | When you need to transform data for visualization. |

Here are some common challenges and how Flows solve them. Don't worry about the details yet—we'll dive deeper into each component below.:

| Component / Scenario | Common Challenge | Flow Solution |
| :--- | :--- | :--- |
| **Batch** 📦 | Dataset returns a `dict` (e.g., `{"x": ..., "y": ...}`), but the pipeline needs `input` and `target`. | Use `batch` to map dictionary keys to `input` and `target` (e.g., `batch: {"input": "x", "target": "y"}`). |
| **Batch in Self-Supervision** | The training step has no `target`. | Use `batch` and just define the `input` (e.g., `batch: ["input"]`). |
| **Loss Function** 📉 | Loss function expects `loss(target, pred)`, but Lighter's default is `loss(pred, target)`. | Use `criterion` to reorder arguments (e.g., `criterion: ["target", "pred"]`). |
| **Metrics** 📏 | A metric needs class indices, but the model outputs probabilities. | Use `metrics` with a transform to apply `torch.argmax` before the metric calculation. |
| **Logging** 📊 | Logger expects RGB images, but the data is grayscale. | Use `logging` with a transform to repeat the channel dimension. |

## Deep Dive

### `batch`

The `batch` component is responsible for extracting `input`, `target`, and an optional `identifier` from each batch produced by your `DataLoader`. This is the first and most common component you'll use.

Lighter needs to know how to get the following from a batch:

| Item | Description |
|-----------|-------------|
| Input | The data fed into the model (e.g., images, text). |
| Target **(optional)** | The ground truth data (e.g., labels, masks) used for loss calculation. |
| Identifier **(optional)** | A unique ID for the data point (e.g., filename, patient ID). |

!!! note
    By default, Lighter assumes the batch is an `(input, target)` tuple:
    `batch: ["input", "target"]`

You can specify `accessors` to handle different batch structures:

| Accessor Type | Description | Example |
| --- | --- | --- |
| **List of Strings** | Access elements by position in a list or tuple batch. | `batch: ["input", "target"]` |
| **Dictionary** | Access elements by key in a dictionary batch. | `batch: {"input": "image", "target": "mask"}` |
| **Callable** | Use a function for complex logic. | `batch: {"target": "$lambda batch: one_hot(batch[1])"}` |

#### Example 1: Dictionary-based Dataset

If your dataset returns a dictionary, you can map the keys to `input`, `target`, and `identifier`.

```yaml
# Problem: Dataset returns a dict, but the model needs tensors.
system:
  flows:
    train:
      batch:
        input: "image"
        target: "mask"
        identifier: "patient_id"
```

#### Example 2: Self-Supervised Learning

In self-supervised learning, you might not have a `target`. You can just omit it from the `batch` definition.

```yaml
# Problem: No targets in this training phase.
system:
  flows:
    train:
      batch: ["input"]
```

For more details, see the [`Flow` documentation](../../reference/flow/#lighter.flow.Flow).

### `criterion`

The `criterion` component acts as a bridge between your model's prediction and your loss function. It allows you to:

1.  **Map** `pred`, `target`, and `input` to the arguments of your loss function.
2.  **Transform** these tensors before they are passed to the loss function.

**Argument Mappers** can be:

| Mapper Type | Description | Example |
| --- | --- | --- |
| **List of Strings** | Map to positional arguments. | `criterion: ["pred", "target"]` |
| **Dictionary** | Map to keyword arguments. | `criterion: {"prediction": "pred", "ground_truth": "target"}` |

**Transforms** are functions applied to the tensors before mapping.

| Transform Type | Description | Example |
| --- | --- | --- |
| **Callable** | A function or list of functions to apply. | `criterion: ["$torch.sigmoid(pred)", "target"]` |

#### Example: Custom Argument Order and Activation

If your loss function expects `(target, pred)` and requires sigmoid on the predictions:

```yaml
# Problem: Loss function has a non-standard signature and needs activated predictions.
system:
  flows:
    train:
      criterion: ["$torch.sigmoid(pred)", "target"]
```

For more details, see the [`Flow` documentation](../../reference/flow/#lighter.flow.Flow).

### `metrics`

The `metrics` component is identical in configuration to the `criterion` component, but for your metrics. You can use it to map and transform `pred`, `target`, and `input` before they are fed into your `torchmetrics` functions.

#### Example: Preparing Predictions for a Metric

If a metric requires class indices from your model's output probabilities:

```yaml
# Problem: Metric expects class indices, not probabilities.
system:
  flows:
    val:
      metrics:
        preds: "$torch.argmax(pred, dim=1)"
        target: "target"
```

For more details, see the [`Flow` documentation](../../reference/flow/#lighter.flow.Flow).

### `logging`

The `logging` component is used to transform `input`, `target`, and `pred` tensors just before they are sent to the logger (e.g., for image visualization in TensorBoard).

#### Example: Visualizing Grayscale Images

If you want to log a single-channel image and your logger expects a 3-channel RGB image:

```yaml
# Problem: Logger expects a 3-channel image, but data is grayscale.
system:
  flows:
    train:
      logging:
        input: "$lambda x: x.repeat(1, 3, 1, 1)" # Convert to 3-channel
```

For more details, see the [`Flow` documentation](../../reference/flow/#lighter.flow.Flow).

### `output`

The `output` component defines what the step should return. This is particularly useful for the `predict` step, where you might want to return the predictions and identifiers.

#### Example: Returning Predictions and Identifiers

```yaml
# Problem: I need to get the predictions and their identifiers from the predict step.
system:
  flows:
    predict:
      output:
        pred: "pred"
        identifier: "identifier"
```

For more details, see the [`Flow` documentation](../../reference/flow/#lighter.flow.Flow).

## Complete Example: Segmentation Pipeline

Here’s how a `Flow` can be used in a complete segmentation pipeline:


```yaml
system:
  flows:
    train:
      # 1. Extract 'image' and 'mask' from the batch dictionary.
      batch:
        input: "image"
        target: "mask"
        identifier: "patient_id"

      # 2. Pass prediction and target to the loss function and apply softmax.
      criterion: ["$torch.nn.functional.softmax(pred, dim=1)", "target"]

      # 3. Pass prediction and target to metrics by name and apply argmax.
      metrics:
        preds: "$torch.argmax(pred, dim=1)"
        target: "target"

    # 4. Reuse the same batch definition for validation.
    val:
      batch: "%system#flows#train#batch"
      # Criterion and Metrics for 'val' would also be defined here.
      metrics: "%system#flows#train#metrics"
```

## Recap and Next Steps

Flows are what make Lighter truly flexible:

✅ **Key Benefits:**

- Handle any data format without code changes
- Connect incompatible components seamlessly
- Transform data at the right pipeline stage
- Debug and monitor data flow easily

🎯 **Best Practices:**

- Keep transforms simple and composable
- Move expensive operations to datasets
- Reuse configurations with YAML anchors

## Related Guides
- [Metrics](metrics.md) - Using Metrics with Flows
- [Writers](writers.md) - Using Writers with Flows
- [Inferers](inferers.md) - Inference-time adaptation
