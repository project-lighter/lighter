Adapters are a powerful Lighter feature that allow data flow customization between different steps. This enables Lighter to handle any task without modifying code.

Examples of when you might need adapters include:

!!! example "Handling Different Batch Structures"

    Often, a batch is simply a tuple of input and target tensors. Other times, a batch may be just the input tensor (e.g. prediction dataset or self-supervised learning). With adapters, you can handle any batch format without changing the core code.


!!! example "Adapting Argument Order"
    
    One loss function may require `(pred, target)` arguments, while another may require `(target, pred)`. The third may require `(input,  pred, target)`. You can specify what should be passed using adapters.

!!! example "Transforming Data"

    You're dealing with grayscale images, but the logger expects RGB images for visualization. Adapters allow you to transform data before such operations.

## Types of Adapters in Lighter

Lighter provides four adapter types, each designed to address specific customization needs:

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

## Recap and Next Steps

Adapters are a cornerstone of Lighter's flexibility and extensibility. By using adapters effectively, you can:

*   Seamlessly integrate Lighter with diverse data formats and batch structures.
*   Customize argument passing to loss functions and metrics.
*   Apply pre-processing and post-processing transforms at various stages of your experiment.
*   Tailor the data that is logged for monitoring and analysis.
*   Create highly specialized and reusable customization components.

With adapters, you adapt Lighter to your specific research needs and build complex, yet well-organized and maintainable deep learning experiment configurations.

Next, explore the [Writers](writers.md) to learn how to save model predictions and outputs to files. For a further read on Lighter's design principles and why we designed adapters, check out the [Design section](../design/overview.md).
