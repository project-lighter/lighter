In deep learning, inference is the process of using a trained model to make predictions on new, unseen data. Within the Lighter framework, **inferers** are specialized components that dictate how this inference process is executed during the validation, testing, and prediction stages of your machine learning experiments.

Inferers are responsible for handling a variety of tasks, including:

*   **Sliding Window Inference:**  Essential for processing large images or volumes that exceed available memory, this technique involves breaking down the input into smaller, overlapping windows, performing inference on each window, and then stitching the results back together.
*   **Model Ensembling:** This involves combining the predictions from multiple models to produce a more accurate and stable final prediction.
*   **Custom Output Processing:**  Inferers can also apply custom post-processing logic to the model's raw outputs. This might include thresholding, applying a softmax function for probabilities, or any other transformation specific to your task.

Lighter seamlessly integrates with MONAI's powerful inferer implementations, providing a strong foundation for your inference workflows. You can readily utilize and customize these pre-built inferers, or you can implement your own custom inferers to address specific inference requirements.

This guide will walk you through the process of implementing and utilizing custom inferers within the Lighter framework, giving you fine-grained control over the inference process in your deep learning experiments.

## Using MONAI Inferers in Lighter

Lighter provides out-of-the-box integration with MONAI's inferers, offering a wide array of pre-built implementations that you can easily incorporate into your projects.

#### Configuration

To use a MONAI inferer, you simply need to configure it within the `system.inferer` section of your `config.yaml` file. Here's an example of how to configure the `SlidingWindowInferer`:

```yaml title="config.yaml"
system:
  inferer:
    _target_: monai.inferers.SlidingWindowInferer  # Specify the inferer class
    roi_size:                         # Region of interest size for each window
    sw_batch_size: 4                               # Batch size for processing windows
    overlap: 0.5                                   # Overlap ratio between windows
```

*   **`_target_`:** This key specifies the fully qualified class name of the inferer you want to use. In this case, it's `monai.inferers.SlidingWindowInferer`.
*   **Inferer-Specific Arguments:** The remaining keys (`roi_size`, `sw_batch_size`, `overlap`) are arguments specific to the `SlidingWindowInferer`. Consult the MONAI documentation for detailed information about the available inferers and their respective arguments.

#### Commonly Used MONAI Inferers

Here are some of the commonly used MONAI inferers:

*   **`monai.inferers.SlidingWindowInferer`:**  Ideal for handling large images or volumes that don't fit into memory.
*   **`monai.inferers.SimpleInferer`:** A basic inferer that directly passes the entire input to the model. Suitable when your input data can fit in memory.
*   **`monai.inferers.EnsembleInferer`:**  Facilitates combining predictions from multiple models.
*   **`monai.inferers.patch_inferer`:**  Designed for patch-based inference strategies.

#### Example: Using `SlidingWindowInferer` for Validation

```yaml title="config.yaml"
system:
  model: #... your model definition...

  inferer:
    _target_: monai.inferers.SlidingWindowInferer
    roi_size:
    sw_batch_size: 8
    overlap: 0.25

  def validation_step(self, batch, batch_idx):
    output = super().validation_step(batch, batch_idx)
    pred = output[Data.PRED]  # Access predictions (processed by the inferer)
    #... rest of your validation logic...
```

In this example, the `SlidingWindowInferer` is configured to process inputs during the validation stage. Lighter automatically incorporates this inferer into the `forward` pass of your `System` (defined in `system.py`). When `self.forward(input)` is called within `validation_step`, Lighter checks if an inferer is configured and if the current mode is 'val', 'test', or 'predict'. If so, it utilizes the inferer to process the input and obtain predictions.

## Implementing a Custom Inferer

While MONAI offers a comprehensive collection of inferers, you may encounter situations where you need to implement custom inference logic. This could be due to:

*   **Advanced Ensembling Strategies:** Implementing ensembling techniques beyond simple averaging.
*   **Highly Specialized Output Processing:**  Tailoring output processing to your unique research problem.

To implement a custom inferer in Lighter, you'll create a Python class that adheres to a specific structure.

### Custom Inferer Class Structure

```python title="my_project/inferers/my_custom_inferer.py"
from typing import Any

import torch
from torch.nn import Module

class MyCustomInferer:
    def __init__(self, arg1, arg2, **kwargs):
        """
        Initialize your custom inferer.

        Args:
            arg1: Custom argument 1.
            arg2: Custom argument 2.
            **kwargs: Additional keyword arguments.
        """
        self.arg1 = arg1
        self.arg2 = arg2
        #... initialize any internal components...

    def __call__(self, inputs: torch.Tensor, network: Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Perform inference using your custom logic.

        Args:
            inputs: Input tensor(s) to the model.
            network: The deep learning model (torch.nn.Module).
            *args: Additional positional arguments (if needed).
            **kwargs: Additional keyword arguments (if needed).

        Returns:
            torch.Tensor: The processed prediction tensor(s).
        """
        # Implement your custom inference logic here
        # This could include:
        #   - Test-time augmentation
        #   - Model ensembling
        #   - Sliding window or patch-based inference
        #   - Any other custom processing

        # Example: Simple forward pass with optional post-processing
        outputs = network(inputs, *args, **kwargs)
        processed_outputs = self.post_process(outputs)
        return processed_outputs

    def post_process(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Optional post-processing of model outputs.

        Args:
            outputs (torch.Tensor): Raw model output tensor(s).

        Returns:
            torch.Tensor: Processed output tensor(s).
        """
        # Implement post-processing logic if needed (e.g., thresholding, softmax)
        return outputs
```

### Key Components

1.  **`__init__`:**
    *   This is the constructor of your inferer class.
    *   It takes any custom arguments that you can define in your `config.yaml`.
    *   Use this method to initialize any internal components or parameters your inferer needs.

2.  **`__call__`:**
    *   This method makes your class callable like a function, enabling it to be used directly for inference.
    *   **Arguments:**
        *   `inputs (torch.Tensor)`: The input tensor(s) to your model.
        *   `network (torch.nn.Module)`: Your deep learning model (equivalent to `self.model` in your `System`).
        *   `*args`, `**kwargs`:  These allow you to pass additional arguments if required, although they are not typically used in inferers.
    *   **Logic:**
        *   This is where you implement your core inference logic.
        *   A common pattern is to perform a forward pass through your `network` using  `outputs = network(inputs)`.
        *   You can integrate various inference techniques here, such as TTA, ensembling, or sliding window inference.
        *   You can also call a `post_process` method to further refine the model's raw outputs.
    *   **Return Value:**
        *   This method must return the processed prediction tensor(s) as a `torch.Tensor`. This output will be used as the `pred` value in your validation, testing, or prediction steps.

3.  **`post_process` (Optional):**
    *   This is an optional method for applying post-processing operations to the model's raw outputs.
    *   You can use it for tasks like thresholding, applying a softmax function, or any other custom processing relevant to your problem.
    *   If no post-processing is required, you can simply return the `outputs` tensor directly.

#### Integrating a Custom Inferer

1.  **Save:** Save your custom inferer class (e.g., `MyCustomInferer`) in a Python file within your project (e.g., `my_project/inferers/my_custom_inferer.py`).

2.  **Configure:**  In your `config.yaml`, specify the inferer within the `system.inferer` section, providing the path to your class and any necessary arguments for its `__init__` method:

    ```yaml title="config.yaml"
    system:
      inferer:
        _target_: my_project.inferers.my_custom_inferer.MyCustomInferer
        arg1: value1
        arg2: value2
    ```

    *   **`_target_`:** Points to your custom inferer class.
    *   **`arg1` and `arg2`:**  Arguments passed to your inferer's `__init__` method.

With this configuration, Lighter will create an instance of your custom inferer and use it during the appropriate stages of your experiment.