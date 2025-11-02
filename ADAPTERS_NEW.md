# Proposal: The Simplified `Flow` Adapter

## 1. Core Philosophy: Convention over Configuration

This proposal refines the `Flow` adapter by embracing a "convention over configuration" philosophy to further simplify its use. We remove the explicit `pack` and `unpack` keys and instead rely on sensible, standard conventions for the outputs of the model and criterion.

The goal is to create the cleanest possible configuration for the most common use cases, while retaining flexibility for advanced scenarios.

## 2. The Simplified `Flow` Schema

The `Flow` adapter still defines the entire step, but the configuration for each stage is now flatter.

-   The `batch` section still defines how to unpack the batch into the context.
-   The `model`, `criterion`, and `metrics` sections now **directly** define the keyword arguments for their respective functions.
-   The `output` section defines the final dictionary that the step returns.
-   **Conventions** are used for intermediate values:
    -   The output of the `model` is **always** stored in the context as `pred`.
    -   The output of the `criterion` is **always** stored in the context as `loss`.

```yaml
system:
  # The `adapters` key is kept for user familiarity, but it now configures Flow objects.
  adapters:
    train: # The Flow adapter for the training mode
      _target_: lighter.flow.Flow

      # Defines how the batch is unpacked into the initial context.
      batch:
        input: 0
        target: 1
        identifier: 2

      # Defines the kwargs for the model.
      # The model's output will be automatically saved to context['pred'].
      model:
        input: "input"

      # Defines the kwargs for the criterion.
      # The criterion's output will be automatically saved to context['loss'].
      criterion:
        pred: "pred"
        target: "target"

      # Defines the kwargs for the metrics.
      metrics:
        preds: "pred"
        target: "target"

      # Defines the final output dictionary of the step.
      output:
        identifier: "identifier"
        input: "input"
        target: "target"
        pred: "pred"
        loss: "loss"
```

## 3. How It Works

The internal logic of the `Flow` adapter becomes more opinionated:

1.  Initialize `context = {}`.
2.  Unpack the `batch` into `context` using the `batch` config.
3.  Create `kwargs` for the model from `context` using the `model` config.
4.  Call `model_output = model(**kwargs)`.
5.  **Convention:** Set `context['pred'] = model_output`.
6.  Create `kwargs` for the criterion from `context` using the `criterion` config.
7.  Call `criterion_output = criterion(**kwargs)`.
8.  **Convention:** Set `context['loss'] = criterion_output`.
9.  Repeat for `metrics`.
10. Use the `output` config to build and return the final output dictionary from the `context`.

## 4. Advanced Example

This design still supports complex scenarios. Let's consider a multi-modal task with non-standard data structures to see how the `Flow` adapter handles it.

**Scenario:**
- **Batch:** A dictionary: `{"image_data": ..., "text_data": ..., "metadata": {"label": ..., "sample_weight": ...}}`.
- **Model:** Takes `image_input` and `text_input`, returns a dictionary `{'logits': ..., 'embedding': ...}`.
- **Criterion:** A function with the signature `custom_loss(y_hat, y, weights)`.

Here is the corresponding `Flow` configuration:

```yaml
system:
  adapters:
    train:
      _target_: lighter.flow.Flow

      # 1. Unpack the complex batch dictionary into the context.
      batch:
        image: "$lambda batch: batch['image_data']"
        text: "$lambda batch: batch['text_data']"
        label: "$lambda batch: batch['metadata']['label']"
        weight: "$lambda batch: batch['metadata']['sample_weight']"

      # 2. Define kwargs for the multi-input model.
      model:
        image_input: "image"
        text_input: "text"

      # 3. Define kwargs for the criterion with non-standard names.
      criterion:
        y_hat: "$lambda ctx: ctx.pred['logits']" # Access 'logits' from the model output dict
        y: "label"
        weights: "weight"

      # 4. Define kwargs for the metrics, using a transformation pipeline.
      metrics:
        predictions:
          - "$lambda ctx: ctx.pred['logits']" # Start with the logits
          - "$lambda logits: torch.argmax(logits, dim=1)" # Apply argmax
        truth: "label"

      # 5. Define the final output of the step.
      output:
        pred: "pred.logits" # Can access nested data for the final output
        loss: "loss"
```

## 5. Evaluation: The Final Design

This simplified `Flow` adapter achieves an excellent balance between simplicity and power. It provides a solid, user-friendly foundation for the new adapter system.

---

## 6. Implementation To-Do List

1.  **File System:**
    -   Delete `src/lighter/adapters.py`.
    -   Create a new file `src/lighter/flow.py` and implement the `Flow` class inside it.

2.  **`src/lighter/__init__.py`:**
    -   Add `from .flow import Flow`.
    -   Add `"Flow"` to the `__all__` list.

3.  **`src/lighter/utils/types/containers.py`:**
    -   Delete the `Train`, `Val`, `Test`, `Predict`, and `Adapters` dataclasses.
    -   Remove the import statement for the old adapter classes.

4.  **`src/lighter/engine/schema.py`:**
    -   Completely rewrite the schema for `system.adapters` to validate the new `Flow` configuration structure. It should expect a dictionary of modes (`train`, `val`, etc.), each containing a dictionary that can be instantiated as a `Flow` object.

5.  **`src/lighter/system.py`:**
    -   **`__init__`**: Change the `adapters` argument to accept the new `Flow` configuration. The `self.adapters` attribute will now store the instantiated `Flow` objects for each mode.
    -   **`_step`**: Rewrite this method entirely. It will now be very simple:
        1.  Get the `Flow` object for the current `self.mode`.
        2.  Execute it: `output = self.flows[self.mode](batch=batch, model=self.model, ...)`.
        3.  Call `self._log_stats` with the loss and metrics from the `output` dictionary.
        4.  Return the `output`.
    -   **`forward`**: This method will be removed. The model call is now managed directly by the `Flow` adapter.
    -   **`_prepare_batch`**: This method will be removed.
    -   **`_calculate_loss`**: This method will be removed.
    -   **`_calculate_metrics`**: This method will be removed.
    -   **`_prepare_output`**: This method will be removed.
