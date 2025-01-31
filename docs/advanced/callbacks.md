# Callbacks

Callbacks in Lighter allow you to customize and extend the training process. You can define custom actions to be executed at various stages of the training loop.

## Freezer Callback
The `Freezer` callback allows you to freeze certain layers of the model during training. This can be useful for transfer learning or fine-tuning.

## Writer Callbacks
Lighter provides writer callbacks to save predictions in different formats. The `FileWriter` and `TableWriter` are examples of such callbacks.

- **FileWriter**: Writes predictions to files, supporting formats like images, videos, and ITK images.
- **TableWriter**: Saves predictions in a table format, such as CSV.

For more details on how to implement and use callbacks, refer to the [PyTorch Lightning Callback documentation](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html).
