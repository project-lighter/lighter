import inspect
import random

from loguru import logger
import torch
import torchvision
from torch.nn import Identity, Module, Sequential


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


def hasarg(callable, arg_name):
    args = inspect.signature(callable).parameters.keys()
    return arg_name in args


def get_name(x, include_module_name=False):
    """Get the name of an object, class or function."""
    name = type(x).__name__ if isinstance(x, object) else x.__name__
    if include_module_name:
        module = type(x).__module__ if isinstance(x, object) else x.__module__
        name = f"{module}.{name}"
    return name


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


def debug_message(mode, input, target, pred, metrics, loss):
    is_tensor_loggable = lambda x: (torch.tensor(x.shape[1:]) < 16).all()
    msg = f"\n----------- Debugging Output -----------\nMode: {mode}"
    for name in ["input", "target", "pred"]:
        tensor = eval(name)
        msg += f"\n\n{name.capitalize()} shape and tensor:\n{tensor.shape}"
        msg += f"\n{tensor}" if is_tensor_loggable(tensor) else "\n*Tensor is too big to log"
    msg += f"\n\nLoss:\n{loss}"
    msg += f"\n\nMetrics:\n{metrics}"
    logger.debug(msg)


def reshape_pred_if_single_value_prediction(pred, target):
    """When the task is to predict a single value, pred and target dimensions often
    mismatch - dataloader returns the value in the (B) shape, while the network
    returns predictions in the (B, 1) shape, where the second dim is redundant.
    """
    if isinstance(pred, torch.Tensor) and target is not None:
        if len(pred.shape) == 2 and len(target.shape) == 1 == pred.shape[1]:
            return pred.flatten()
    return pred


def dot_notation_setattr(obj, attr, value):
    # https://gist.github.com/alixedi/4695abcd259d1493ac9c
    """Set object's attribute. May use dot notation.

    >>> class C(object): pass
    >>> a = C()
    >>> a.b = C()
    >>> a.b.c = 4
    >>> rec_setattr(a, 'b.c', 2)
    >>> a.b.c
    2
    """
    if '.' not in attr:
        setattr(obj, attr, value)
    else:
        splitted = attr.split('.')
        dot_notation_setattr(getattr(obj, splitted[0]), '.'.join(splitted[1:]), value)

def replace_layer_with(model: Module, layer_name: str, new_layer: Module) -> Module:
    """Replaces the specified layer of the network with another layer.

    Args:
        model (Module): PyTorch model to be edited
        layer_name (string): Name of the layer which will be replaced.
            Dot-notation supported, e.g. "layer10.fc". 

    Returns:
        Module: PyTorch model with the new layer set at the specified location.
    """
    dot_notation_setattr(model, layer_name, new_layer)
    return model

def replace_layer_with_identity(model: Module, layer_name: str) -> Module:
    """Replaces any layer of the network with an Identity layer.
    Useful for removing the last layer of a network to be used as a backbone
    of an SSL model. 

    Args:
        model (Module): PyTorch model to be edited
        layer_name (string): Name of the layer which will be replaced with an
            Identity function. Dot-notation supported, e.g. "layer10.fc". 

    Returns:
        Module: PyTorch model with Identity layer at the specified location.
    """
    return replace_layer_with(model, layer_name, Identity())


def remove_last_layer_sequentially(model: Module()) -> Sequential:
    """Removes the last layer of a network and returns it as an nn.Sequential model.
    Useful when a network is to be used as a backbone of an SSL model.

    Args:
        model (Module): PyTorch model object.

    Returns:
        Sequential: PyTorch Sequential model with the last layer removed.
    """
    return Sequential(*list(model.children())[:-1])
