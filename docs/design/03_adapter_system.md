# Understanding Lighter's Flexible Adapter System

## Introduction to Adapters

Adapters in Lighter offer a flexible way to customize data handling and argument passing, acting as intermediaries to adapt data and function arguments without altering Lighter's core code. This is essential for:

*   **Data Flexibility**: Handling diverse data formats.
*   **Custom Processing**: Applying transformations at different workflow stages.
*   **Argument Mapping**: Matching function arguments to loss, metric, or other component inputs.
*   **Modularity**: Separating customization logic for better code organization and maintainability.

Lighter includes built-in adapter types and supports custom adapters for specific needs.

## Types of Adapters in Lighter

Lighter includes the following adapter types:

1.  **`BatchAdapter`**:

    *   **Purpose**: The `BatchAdapter` extracts essential components from a data batch: **input**, **target**, and **identifier**. The identifier can be used to track individual samples or for specific metrics/logging. It standardizes batch structure across datasets.
    *   **Mechanism**: It uses **accessors** to retrieve data from batches. Accessors can be:
        *   **Index (int)**: For list-like or tuple-like batches (e.g., `batch[0]` for input).
        *   **Key (str)**: For dictionary-like batches (e.g., `batch["image"]` for input).
        *   **Callable (function)**: For custom batch structures or complex extraction logic (e.g., `lambda batch: batch["data"]["image"]`).
    *   **Use Cases**:
        *   Handling batches from various datasets with different structures.
        *   Extracting data from nested batch structures.
        *   Providing a consistent input format for the training loop, regardless of the dataset.

    **Example Configuration (`config.yaml`)**:

    ```yaml
    system:
      adapters:
        train: # Adapters for 'train' stage
          batch: # BatchAdapter configuration
            _target_: lighter.adapters.BatchAdapter
            input_accessor: 0 # Input is at index 0 of the batch
            target_accessor: 1 # Target is at index 1
            identifier_accessor: None # No identifier in this dataset
    ```

    In this example, the `BatchAdapter` for the `train` stage is configured to extract:

    *   **Input**: From index `0` of the batch.
    *   **Target**: From index `1` of the batch.
    *   **Identifier**: No identifier is extracted (`identifier_accessor: None`).

2.  **`CriterionAdapter`**:

    *   **Purpose**: The `CriterionAdapter` adapts arguments and applies transformations to **input**, **target**, and **prediction** data before passing them to the **criterion** (loss function).
    *   **Mechanism**: It performs two main functions:
        *   **Argument Mapping**: Maps the `input`, `target`, and `prediction` to the expected arguments of the criterion function. You can specify argument positions (integers) or argument names (strings).
        *   **Data Transformation**: Applies transformations to `input`, `target`, and `prediction` data using specified transform functions.
    *   **Use Cases**:
        *   Using loss functions with different argument orders than Lighter's default (`pred`, `target`).
        *   Applying preprocessing or postprocessing steps to data before calculating the loss (e.g., converting predictions to class labels, normalizing targets).
        *   Adapting data formats to match the criterion's input requirements.

    **Example Configuration (`config.yaml`)**:

    ```yaml
    system:
      adapters:
        train: # Adapters for 'train' stage
          criterion: # CriterionAdapter configuration
            _target_: lighter.adapters.CriterionAdapter
            pred_argument: 0 # Prediction is the first argument to criterion
            target_argument: 1 # Target is the second argument
            pred_transforms: # Transformations for predictions
              - _target_: torch.argmax # Apply argmax to predictions
                dim: 1 # Along dimension 1 (class dimension)
    ```

    In this example, the `CriterionAdapter` for the `train` stage is configured to:

    *   Pass **prediction** as the **first positional argument** to the criterion function.
    *   Pass **target** as the **second positional argument**.
    *   Apply `torch.argmax(dim=1)` transformation to the **prediction** before passing it to the criterion.

3.  **`MetricsAdapter`**:

    *   **Purpose**: The `MetricsAdapter` is similar to `CriterionAdapter` but is used for **metrics**. It adapts arguments and applies transformations to **input**, **target**, and **prediction** data before passing them to the evaluation **metrics**.
    *   **Mechanism**: It provides the same argument mapping and data transformation capabilities as `CriterionAdapter`, but for metrics functions or classes.
    *   **Use Cases**:
        *   Using metrics with different argument orders.
        *   Applying preprocessing or postprocessing steps to data before calculating metrics.
        *   Adapting data formats for metrics calculation.

    **Example Configuration (`config.yaml`)**:

    ```yaml
    system:
      adapters:
        val: # Adapters for 'val' stage
          metrics: # MetricsAdapter configuration
            _target_: lighter.adapters.MetricsAdapter
            input_argument: "inputs" # Input is passed as keyword argument 'inputs'
            target_argument: "labels" # Target as keyword argument 'labels'
            pred_argument: "outputs" # Prediction as keyword argument 'outputs'
            pred_transforms: # Transformations for predictions
              - _target_: torch.sigmoid # Apply sigmoid to predictions
    ```

    In this example, the `MetricsAdapter` for the `val` stage is configured to:

    *   Pass **input** as the keyword argument `inputs` to the metric function.
    *   Pass **target** as the keyword argument `labels`.
    *   Pass **prediction** as the keyword argument `outputs`.
    *   Apply `torch.sigmoid()` transformation to the **prediction** before passing it to the metric.

4.  **`LoggingAdapter`**:

    *   **Purpose**: The `LoggingAdapter` applies transformations specifically for **logging** purposes. It allows you to preprocess or format **input**, **target**, and **prediction** data before they are logged by Lighter.
    *   **Mechanism**: It uses transformation functions to modify the data before logging. It does **not** handle argument mapping, as logging typically involves recording data as-is.
    *   **Use Cases**:
        *   Formatting data for better readability in logs (e.g., converting tensors to NumPy arrays or lists).
        *   Filtering sensitive or irrelevant data from logs.
        *   Applying specific transformations to visualize data in logging outputs, such as formatting images.

    **Example Configuration (`config.yaml`)**:

    ```yaml
    system:
      adapters:
        train: # Adapters for 'train' stage
          logging: # LoggingAdapter configuration
            _target_: lighter.adapters.LoggingAdapter
            input_transforms: # Transformations for input data in logs
              - _target_: lambda x: x.cpu().numpy() # Convert input tensor to NumPy array
            pred_transforms: # Transformations for prediction data in logs
              - _target_: lambda x: torch.argmax(x, dim=1).cpu().numpy() # Get class labels and convert to NumPy
            target_transforms: # Transformations for target data in logs
              - _target_: torchvision.transforms.ToPILImage # Convert target tensor to PIL Image for image logging
    ```

    In this example, the `LoggingAdapter` for the `train` stage is configured to:

    *   Convert **input** tensors to NumPy arrays before logging.
    *   Convert **prediction** tensors to class label NumPy arrays (using `argmax`) before logging.
    *   Convert **target** tensors to PIL Images for image logging.

## Creating Custom Adapters (Advanced)

While Lighter's built-in adapters cover many common use cases, you might need to create **custom adapters** for highly specialized scenarios. To create a custom adapter:

1.  **Define a new Python class** that inherits from one of Lighter's base adapter classes (e.g., `_ArgumentsAdapter`, `_TransformsAdapter`, or combine both by inheriting from `_ArgumentsAndTransformsAdapter`).
2.  **Implement the `__call__` method** in your custom adapter class. This method will contain your custom logic for data adaptation and/or argument mapping.
3.  **Configure your custom adapter** in the `config.yaml` file, specifying the `_target_` path to your custom adapter class and any necessary arguments.

**Example: Custom Adapter for a Different Argument Order in Loss Function**

If you have a loss function that expects arguments in a different order (e.g., `target`, `pred` instead of default `pred`, `target`), create a custom adapter:

```python title="my_project/adapters.py"
from lighter.adapters import CriterionAdapter

class MyLossAdapter(CriterionAdapter): # Inherit from CriterionAdapter
    def __call__(self, criterion, input, target, pred):
        return super().__call__(criterion, input, target, pred) # No changes needed in __call__
```

**Configuration in `config.yaml`**:

```yaml title="config.yaml"
system:
  adapters:
    train:
      criterion:
        _target_: my_project.adapters.MyLossAdapter # Path to custom adapter
        target_argument: 0 # Target is the first argument
        pred_argument: 1 # Prediction is the second argument
  criterion:
    _target_: my_project.losses.MyLossFunction # Your custom loss
```

Here, `MyLossAdapter` configures `target_argument` and `pred_argument` to match `MyLossFunction`'s expected order. No changes to the adapter's `__call__` are needed as we are only reconfiguring argument order.

## Benefits of the Adapter System

*   **Flexibility**: Adapters enable Lighter to seamlessly integrate with diverse datasets, models, loss functions, and metrics, regardless of their input/output formats or argument conventions.
*   **Modularity and Reusability**: Adapter logic is encapsulated in separate classes, promoting code modularity and reusability. You can reuse adapters across different experiments or projects.
*   **Maintainability**: By separating data handling and customization logic into adapters, the core Lighter framework remains clean and maintainable.
*   **Extensibility**: The adapter system is highly extensible. You can easily add new adapter types or customize existing ones to meet evolving project requirements.

## Recap: Adapting Lighter to Your Needs with Adapters

Lighter's adapter system is a powerful mechanism for tailoring the framework to your specific deep learning tasks. By using built-in adapters or creating custom ones, you can effectively handle diverse data formats, customize data processing pipelines, and seamlessly integrate your own components. Adapters contribute significantly to Lighter's flexibility, modularity, and extensibility, making it a versatile tool for a wide range of deep learning experiments.

Next, delve into the [Dynamic Module Loading Design](../design/04_dynamic_module_loading.md) to understand how Lighter handles dynamic imports for custom modules, or return to the [Design section](../design/01_overview.md) for more conceptual documentation. You can also refer back to the [How-To guides section](../how-to/01_custom_project_modules.md) for practical problem-solving guides or the [Tutorials section](../tutorials/01_configuration_basics.md) for end-to-end examples.
