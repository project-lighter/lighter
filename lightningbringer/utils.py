import random

import torch
import torchvision


def import_attr(module_attr):
    """Import using dot-notation string, e.g., 'torch.nn.Module'.

    Args:
        module_attr (str): dot-notation path to the attribute.

    Returns:
        Any: imported attribute.
    """
    # Split module from attribute name
    module, attr = module_attr.rsplit(".", 1)
    # Import the module
    module = __import__(module, fromlist=[attr])
    # Get the attribute from the module
    return getattr(module, attr)


def get_name(x):
    """Get the name of an object, class or function."""
    return type(x).__name__ if isinstance(x, object) else x.__name__


def wrap_into_list(x):
    """Wrap the input into a list if it is not a list. If it is a None, return an empty list."""
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if x is None:
        return []
    return [x]


def collate_fn_replace_corrupted(batch, dataset):
    """Collate function that allows to replace corrupted examples in the batch.
    It expect that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are replaced with other, randomly-selected, examples.

    Args:
        batch (torch.Tensor): batch from the DataLoader.
        dataset (torch.utils.data.Dataset): dataset that the DataLoader is passing through.
            Needs to be fixed in place with functools.partial before passing it to DataLoader's
            'collate_fn' option as 'collate_fn' should only have a single argument - batch.
            E.g.:
                collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=dataset)
                loader = DataLoader(dataset, ..., collate_fn=collate_fn)

    Returns:
        torch.Tensor: batch with new examples instead of corrupted ones.
    """
    # Idea from https://stackoverflow.com/a/57882783
    original_batch_len = len(batch)
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    filtered_batch_len = len(batch)
    # Num of corrupted examples
    num_corrupted = original_batch_len - filtered_batch_len
    if num_corrupted > 0:
        # Replace a corrupted example with another randomly selected example
        batch.extend([dataset[random.randint(0, len(dataset))] for _ in range(num_corrupted)])
        # Recursive call to replace the replacements if they are corrupted
        return collate_fn_replace_corrupted(batch, dataset)
    # Finally, when the whole batch is fine, return it
    return torch.utils.data.dataloader.default_collate(batch)


def preprocess_image(image):
    """Preprocess the image for logging. If it is a batch of multiple images,
    it will create a grid image of them. In case of 3D, a single image is displayed
    with slices stacked vertically, while a batch as a grid where each column is
    a different 3D image.

    Args:
        image (torch.Tensor): 2D or 3D image tensor.

    Returns:
        torch.Tensor: image ready for logging.
    """
    image = image.detach().cpu()
    # 3D image (NCDHW)
    has_three_dims = image.ndim == 5
    if has_three_dims:
        shape = image.shape
        image = image.view(shape[0], shape[1], shape[2] * shape[3], shape[4])
    # If more than one image, create a grid
    if image.shape[0] > 1:
        nrow = image.shape[0] if has_three_dims else 8
        image = torchvision.utils.make_grid(image, nrow=nrow)
    return image


def debug_message(mode, input, label, pred, metrics, loss):
    is_tensor_loggable = lambda x: (torch.tensor(x.shape[1:]) < 16).all()
    msg = f"\n----------- Debugging Output -----------\nMode: {mode}"
    for name in ["input", "label", "pred"]:
        tensor = eval(name)
        msg += f"\n\n{name.capitalize()} shape and tensor:\n{tensor.shape}"
        msg += f"\n{tensor}" if is_tensor_loggable(tensor) else "\n*Tensor is too big to log"
    msg += f"\n\nLoss:\n{loss}"
    msg += f"\n\nMetrics:\n{metrics}"
    logger.debug(msg)


def reshape_pred_if_single_value_prediction(pred, label):
    """When the task is to predict a single value, pred and label dimensions often
    mismatch - dataloader returns the value in the (B) shape, while the network
    returns predictions in the (B, 1) shape, where the second dim is redundant.
    """
    if isinstance(pred, torch.Tensor) and label is not None:
        if len(pred.shape) == 2 and len(label.shape) == 1 == pred.shape[1]:
            return pred.flatten()
    return pred