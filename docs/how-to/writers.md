# Writers: Save Your Results Like a Pro

Writers are your data persistence layer—they capture model outputs and save them in formats ready for analysis, visualization, or deployment.

## Quick Start 🚀

```yaml
# Save predictions as images
trainer:
    callbacks:
        - _target_: lighter.callbacks.FileWriter
          path: "outputs/predictions"
          writer: "image"  # PNG for 2D, MP4 for 3D

# Save metrics to CSV
trainer:
    callbacks:
        - _target_: lighter.callbacks.TableWriter
          path: "outputs/metrics.csv"
```

## Writer Types at a Glance

| Writer | Purpose | Output Format | Best For |
|--------|---------|---------------|----------|
| **FileWriter** | Save predictions/tensors | NIfTI, PNG, MP4, NPY | Images, volumes, arrays |
| **TableWriter** | Save tabular data | CSV | Metrics, statistics, results |

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
2.  Applies `logging` transforms (if configured in the `Flow`).
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

## Custom Writer Example

```python
# my_project/writers/visualization_writer.py
import matplotlib.pyplot as plt
from pathlib import Path

class VisualizationWriter:
    """Save comparison plots of input, target, and prediction."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)

    def write_comparison(self, input_img, target, prediction, identifier):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(input_img.cpu().numpy().transpose(1, 2, 0))
        axes[0].set_title("Input")

        axes[1].imshow(target.cpu().numpy(), cmap='tab20')
        axes[1].set_title("Ground Truth")

        axes[2].imshow(prediction.argmax(0).cpu().numpy(), cmap='tab20')
        axes[2].set_title("Prediction")

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(self.path / f"{identifier}_comparison.png")
        plt.close()
```

## Quick Reference 📄

### FileWriter Formats
```yaml
# Medical imaging
writer: "itk_nifti"     # .nii.gz files
writer: "itk_nrrd"      # .nrrd files
writer: "itk_seg_nrrd"  # Segmentation masks

# Standard formats
writer: "tensor"        # NumPy .npy files
writer: "image"         # PNG (2D) or MP4 (3D)
writer: "video"         # MP4 for time series

# Custom
writer: my_project.writers.custom_writer
```

### TableWriter Patterns
```python
# Return dict from step methods for TableWriter
def validation_step(self, batch, batch_idx):
    # ... compute metrics ...
    return {
        "patient_id": batch["id"],
        "dice_score": dice,
        "loss": loss.item(),
        "prediction_confidence": pred.max()
    }
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Permission denied** | Create output directory: `Path("outputs").mkdir(exist_ok=True)` |
| **Out of disk space** | Use compression or write less frequently |
| **Slow writing** | Reduce precision to FP16 or use async writing |

## Recap and Next Steps

✅ **You've Learned:**
- Use FileWriter for predictions and tensors
- Use TableWriter for metrics and results
- Create custom writers for special needs
- Optimize writing for performance

🎯 **Best Practices:**
- Organize outputs hierarchically
- Use compression for large outputs
- Consider async writing for speed
- Save metadata with predictions

💡 **Pro Tip:** Always save enough information to reproduce your results!

## Related Guides
- [Flows](flows.md) - Transform before writing
- [Inferers](inferers.md) - Write inference results
