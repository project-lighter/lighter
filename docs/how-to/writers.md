Lighter writers are callbacks for saving model predictions and outputs to files during validation, testing, and prediction. They offer a standardized, extensible way to persist experiment results for analysis and visualization.

Lighter offers two main writer types:

1.  **`FileWriter`**: Saves tensors (predictions, images) to files (NIfTI, NRRD, PNG, MP4, NumPy).
2.  **`TableWriter`**: Saves tabular data (metrics, aggregated predictions) to CSV files.

This guide explains how to use and extend `FileWriter` and `TableWriter` in Lighter to effectively manage experiment outputs.

## Using `FileWriter`

`FileWriter` callback saves tensors to files, supports various formats, and is customizable.

**Configuration**:

Configure `FileWriter` in `config.yaml` within `trainer.callbacks` section:

```yaml title="config.yaml"
trainer:
  callbacks:
    - _target_: lighter.callbacks.FileWriter # Use the FileWriter callback
      path: "outputs/predictions"          # Directory to save output files
      writer: "itk_nifti"                  # Writer function to use (ITK-NIfTI format)
```

*   **`_target_: lighter.callbacks.FileWriter`**: Specifies that you want to use the `FileWriter` callback.
*   **`path: "outputs/predictions"`**: Defines the directory where the output files will be saved. Lighter will create this directory if it doesn't exist.
*   **`writer: "itk_nifti"`**: Specifies the writer function to be used for saving tensors. In this example, we use `"itk_nifti"`, which saves tensors in the ITK-NIfTI format (commonly used for medical images).

**Built-in Writer Functions**:

`FileWriter` has built-in writer functions for different formats:

*   **`"tensor"`**: Raw NumPy `.npy` files (general tensor saving).
*   **`"image"`**: Images (PNG for 2D, MP4 for 3D animation).
*   **`"video"`**: Videos (MP4 for 4D tensor time-series).
*   **`"itk_nrrd"`**: NRRD files (ITK library, medical imaging).
*   **`"itk_seg_nrrd"`**: NRRD segmentation mask files (ITK).
*   **`"itk_nifti"`**: NIfTI files (ITK library, medical imaging).

**Usage**:

Once configured, `FileWriter` is used by Lighter in validation, test, and predict stages (if enabled).

In these stages, per batch, `FileWriter`:

1.  Receives `pred` tensor from `predict_step`, `validation_step`, or `test_step`.
2.  Applies `LoggingAdapter` transforms (if configured).
3.  Uses writer function (e.g., `"itk_nifti"`) to save `pred` tensor to file in `path` dir.
4.  Names file using batch `identifier` (if available) or generates unique name.

**Example: Saving Segmentation Predictions in NIfTI Format**

```yaml title="config.yaml"
trainer:
  callbacks:
    - _target_: lighter.callbacks.FileWriter
      path: "outputs/segmentations"
      writer: "itk_nifti" # Save as NIfTI files

system:
  # ... (other system configurations) ...
  dataloaders:
    val:
      dataset:
        _target_: monai.datasets.DecathlonDataset
        task: "Task09_Spleen"
        root: "data/"
        section: "validation"
        transform: # ... (data transforms) ...
      batch_size: 1
```

Example config: `FileWriter` saves segmentation predictions during validation stage as NIfTI files in `outputs/segmentations` dir. Filenames use validation dataset identifiers (e.g., patient IDs).

## Extending `FileWriter` with Custom Writers

Extend `FileWriter` by creating custom writer functions or classes for specific needs.

**1. Create Custom Writer Function**:

Define a custom writer function with two arguments:

*   `path`: Full file path for saving tensor (filename & extension).
*   `tensor`: PyTorch tensor to save.

**Example: Custom Writer Function for Text Files**

```python title="my_project/writers/my_custom_writer.py"
import torch
import numpy as np

def write_tensor_as_text(path: str, tensor: torch.Tensor):
    """Saves tensor to text file."""
    tensor_numpy = tensor.cpu().numpy() # Convert to NumPy array
    np.savetxt(path, tensor_numpy)      # Save as text file
```

**2. Register Custom Writer Function**:

Register custom writer function in `config.yaml` to use with `FileWriter`:

```yaml title="config.yaml"
trainer:
  callbacks:
    - _target_: lighter.callbacks.FileWriter
      path: "outputs/text_tensors"
      writer: my_project.writers.my_custom_writer.write_tensor_as_text # Path to custom writer
```

*   **`writer`**: Path to custom writer function. Replace `"my_project.writers.my_custom_writer"` with your module path.

**3. Create Custom Writer Class (Advanced)**:

For complex logic or stateful writers, create a custom class inheriting from `lighter.callbacks.writer.BaseWriter`.

**Example: Custom Writer Class for Tensors with Metadata**

```python title="my_project/writers/my_custom_writer_class.py"
from lighter.callbacks.writer import BaseWriter
import torch
import json
import os

class MyCustomClassWriter(BaseWriter):
    @property
    def writers(self):
        return {"tensor_with_metadata": self.write_tensor_with_metadata} # Register writer function

    def write(self, tensor: torch.Tensor, identifier: str):
        """Main write method called by FileWriter."""
        path = os.path.join(self.path, f"{identifier}.json") # Define output path
        self.write_tensor_with_metadata(path, tensor, identifier=identifier) # Call writer function

    def write_tensor_with_metadata(self, path: str, tensor: torch.Tensor, identifier: str):
        """Saves tensor to JSON file with metadata."""
        metadata = {
            "identifier": identifier,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "timestamp": datetime.datetime.now().isoformat()
        }
        data = {
            "metadata": metadata,
            "data": tensor.cpu().numpy().tolist() # Convert tensor data to list
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=4) # Save data+metadata to JSON
```

`MyCustomClassWriter` class example:

*   Inherits from `BaseWriter`.
*   `write_tensor_with_metadata`: Saves tensors to JSON with metadata (shape, dtype, timestamp).
*   Registers writer function in `writers` property with key `"tensor_with_metadata"`.
*   Overrides `write` method for filename generation and calling custom writer function.

**4. Use Custom Writer Class in `config.yaml`**:

Use custom writer class by specifying class module path and registered writer key in `config.yaml`:

```yaml title="config.yaml"
trainer:
  callbacks:
    - _target_: lighter.callbacks.FileWriter
      path: "outputs/metadata_tensors"
      writer: my_project.writers.my_custom_writer_class.MyCustomClassWriter.tensor_with_metadata # Custom class writer
```

*   **`writer`**: Path to custom writer class and registered writer key (`"tensor_with_metadata"`).

## Using `TableWriter`

`TableWriter` callback saves tabular data to CSV files, useful for logging metrics or aggregated predictions.

**Configuration**:

Configure `TableWriter` in `config.yaml` within `trainer.callbacks` section:

```yaml title="config.yaml"
trainer:
  callbacks:
    - _target_: lighter.callbacks.TableWriter # Use the TableWriter callback
      path: "outputs/metrics.csv"          # Path to save CSV file
```

*   **`_target_: lighter.callbacks.TableWriter`**: Use `TableWriter` callback.
*   **`path: "outputs/metrics.csv"`**: CSV file path for saving tabular data.

**Usage**:

To use `TableWriter`, return a dictionary from `validation_step`, `test_step`, or `predict_step`. `TableWriter` saves key-value pairs from dict as CSV rows.

**Example: Logging Metrics to CSV using `TableWriter`**

```python title="config.yaml"
trainer:
  callbacks:
    - _target_: lighter.callbacks.TableWriter
      path: "outputs/metrics.csv" # Save metrics to CSV

system:
  metrics:
    val:
      - _target_: torchmetrics.Accuracy
      - _target_: torchmetrics.DiceCoefficient

  def validation_step(self, batch, batch_idx):
    output = super().validation_step(batch, batch_idx) # Call base validation step
    metrics = output[Data.METRICS] # Get computed metrics
    self.log_dict(metrics)        # Log metrics for display
    return metrics                # Return metrics dictionary for TableWriter
```

Example: `TableWriter` saves data to `outputs/metrics.csv`. In `validation_step`:

*   Call base class `validation_step` to compute metrics.
*   Extract metrics from `output` dict.
*   Return `metrics` dict from `validation_step`.

`TableWriter` captures dict from `validation_step`/`test_step`/`predict_step`, saves as CSV rows. Dict keys become CSV column headers.

## Extending `TableWriter` (Advanced)

Like `FileWriter`, extend `TableWriter` with custom writer classes for specialized table writing. See `lighter/callbacks/writer/table.py` for details.

## Recap and Next Steps

Lighter writers are key for saving/managing experiment outputs. Use `FileWriter` and `TableWriter`, extend with custom writers as needed to:

*   Save model predictions in various formats for visualization/analysis.
*   Log metrics/tabular data to CSV for experiment tracking/reporting.
*   Create custom output saving logic.

Writers provide a complete solution for running DL experiments and capturing/utilizing valuable outputs.

Next, explore [Freezers](freezers.md) for freezing model layers.
