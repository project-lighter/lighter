# Taken from Detectron 2, licensed under Apache 2.0.
# https://github.com/facebookresearch/detectron2/blob/989f52d67d05445ccd030d8f13d6cc53e297fb91/detectron2/utils/comm.py
# Changes:
# - removed half of the functions
# - init_distributed()
# - `reduce` and `all_reduce` for various datatypes
# - `shared_random_seed` uses `torch.distributed.broadcast` instead of `all_gather` from Detectron2.

from loguru import logger
import os
import functools

import torch
import numpy as np


def init_distributed():
    """Initialize distributed mode if ran with `torch.distributed.launch --use_env`"""
    if os.environ.get('WORLD_SIZE', None):
        num_gpu = int(os.environ.get('WORLD_SIZE', 1))
        if num_gpu > 1:
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            synchronize()
            logger.info(f'Number of GPUs available in world: {num_gpu}.')
        else:
            raise ValueError("Distributed ON but but running single process.")


def synchronize():
    """
    Synchronize processes between GPUs. Wait until all devices are available.
    Function returns nothing in a non-distributed setting too.
    """
    if not torch.distributed.is_available():
        logger.info('torch.distributed: not available.')
        return

    if not torch.distributed.is_initialized():
        logger.info('torch.distributed: not initialized.')
        return

    if torch.distributed.get_world_size() == 1:
        logger.info('torch distributed: world size is 1')
        return

    torch.distributed.barrier()


def get_rank() -> int:
    """
    Get rank of the process, even when torch.distributed is not initialized.
    Returns
    -------
    int
    """
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0

    return torch.distributed.get_rank()


def get_local_rank() -> int:
    """
    Get rank of the process, even when torch.distributed is not initialized.
    Returns
    -------
    int
    """
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0

    return int(os.environ['LOCAL_RANK'])


def get_world_size() -> int:
    """
    Get number of compute device in the world, returns 1 in case multi device is not initialized.
    Returns
    -------
    int
    """
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_backend_compatible_device():
    """Use device that is compatible with the backend (e.g. nccl does not support CPU tensors).
    Used when initializing placeholder tensors before performing communication operations.
    """
    return torch.device('cuda' if torch.distributed.get_backend() == 'nccl' else 'cpu')


def shared_random_seed() -> int:
    """
    All workers must call this function, otherwise it will deadlock.
    Returns
    -------
    A random number that is the same across all workers. If workers need a shared RNG, 
    they can use this shared seed to create one.
    """
    # torch.Generator advises to use a high values as seed, hence 2**31
    # The seed is reproducible when torch seed is set with `torch.manual_seed()`
    seed = torch.randint(2**31, (1,))
    if torch.distributed.is_initialized():
        device = get_backend_compatible_device()
        seed = seed.to(device)
        torch.distributed.broadcast(seed, 0)
    return int(seed)


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if torch.distributed.get_backend() == "nccl":
        return torch.distributed.new_group(backend="gloo")
    return torch.distributed.group.WORLD


# ------------------ Gather -----------------------


def gather(input_data):
    # TODO: Assert for PyTorch version since `gather_object` present from 1.8.0
    if get_world_size() < 2:
        return input_data

    input_data = move_to(input_data, torch.device('cpu'))
    group = _get_global_gloo_group()

    if torch.distributed.get_rank(group=group) == 0:
        gather_list = [None for _ in range(get_world_size())]
        torch.distributed.gather_object(input_data, gather_list, dst=0, group=group)
        return gather_list
    else:
        torch.distributed.gather_object(input_data, dst=0, group=group)
        return input_data


# ------------ Reduce and All Reduce --------------


def reduce(input_data, average=False, all_reduce=False):
    """
    Interface function for performing reduce on any type of 
    data [int, float, tensor, dict, list, tuple] by summing or 
    averaging the value(s) using one of the methods:

    (1) rank 0 reduce (torch.distributed.reduce):
        communicates the sum or average of
        all processes to the process of rank 0 only

    (2) all reduce (torch.distributed.all_reduce)
        communicates the sum or average of
        all processes to each process

    Parameters:
        input_dict (int, float, tensor, dict, list, tuple) -- data for reduction
        average (bool) -- true if the results should be averaged
        all_reduce (bool) -- true if communicating the reduced value to all processes,
                             otherwise process of rank 0 only

    Returns the sum or average of the input data from across processes.
    """
    if get_world_size() < 2:
        return input_data

    device = get_backend_compatible_device()
    with torch.no_grad():
        if isinstance(input_data, torch.Tensor):
            reduced_data = reduce_tensor(input_data, average, all_reduce, device)

        elif isinstance(input_data, dict):
            reduced_data = reduce_dict(input_data, average, all_reduce, device)

        elif isinstance(input_data, (list, tuple)):
            reduced_data = reduce_list_tuple(input_data, average, all_reduce, device)

        elif isinstance(input_data, (float, int)):
            reduced_data = reduce_int_float(input_data, average, all_reduce, device)

        else:
            data_type = str(type(input_data))
            raise NotImplementedError(f"Reduction on data type `{data_type}` is not implemented.")
    return reduced_data


def reduce_tensor(tensor, average, all_reduce, device):
    """Reduce a tensor"""
    tensor = tensor.clone().to(device)
    if all_reduce:
        torch.distributed.all_reduce(tensor)
    else:
        torch.distributed.reduce(tensor, dst=0)

    if average and (get_rank() == 0 or all_reduce):
        tensor /= get_world_size()
    return tensor


def reduce_int_float(input_value, average, all_reduce, device):
    """Reduce an integer or float by converting it to tensor, 
    performing reduction and, finally, casting it back to the initial type. 
    """
    data_type = type(input_value)  # save the original data type
    tensor = torch.Tensor([input_value])  # convert to tensor
    tensor = reduce_tensor(tensor, average, all_reduce, device)
    reduced_value = data_type(tensor.item())  # cast back to int or float
    return reduced_value


def reduce_dict(input_dict, average, all_reduce, device):
    """ Reduce a dict by extracting all of its values into a tensor and communicating it.
    Returns a dict with the same fields as input_dict, after reduction. If its values were 
    int or float, they are converted to tensors.
    """
    names = []
    values = []
    for k in input_dict.keys():
        names.append(k)
        value = input_dict[k]
        # if float or integer and not tensor, convert to tensor
        if is_not_tensor(value):
            if is_float_or_int(value):
                value = torch.Tensor([value])
            elif is_numpy_scalar(value):
                value = torch.Tensor([value.item()])
            else:
                raise NotImplementedError("Dictionary reduction supported only if its values \
                                           are tensors, numpy scalars floats or integers.")
        values.append(value)
    values = torch.stack(values, dim=0).to(device)  # convert the list of tensors to a single tensor

    if all_reduce:
        torch.distributed.all_reduce(values)
    else:
        torch.distributed.reduce(values, dst=0)

    if average and (get_rank() == 0 or all_reduce):
        values /= get_world_size()

    reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def reduce_list_tuple(input_data, average, all_reduce, device):
    """ Reduce a list or tuple whose elements are either tensors, floats or integers.
    Returns reduced list/tuple with its elements as tensors.
    """
    data_type = type(input_data)  # save the original data type
    # convert tuple/list values to tensors if they are floats or integers
    for i in range(len(input_data)):
        value = input_data[i]
        if is_not_tensor(value):
            if is_float_or_int(value):
                input_data[i] = torch.Tensor([value])
            elif is_numpy_scalar(value):
                input_data[i] = torch.Tensor([value.item()])
            else:
                raise NotImplementedError("List/tuple reduction supported only if"
                                          " its values are tensors, floats or integers.")
    # Convert list/tuple of tensors to a single tensor
    values = torch.stack(input_data, dim=0).to(device)

    if all_reduce:
        torch.distributed.all_reduce(values)
    else:
        torch.distributed.reduce(values, dst=0)

    if average and (get_rank() == 0 or all_reduce):
        values /= get_world_size()

    # Cast it back to tuple or list
    reduced_sequence = data_type(values)
    return reduced_sequence


# ------------------ Helpers -----------------------


def move_to(obj, device):
    """Move any combination of list or dict containing tensors to the specific device."""
    if not isinstance(obj, (torch.Tensor, dict, list)):
        return obj
    elif torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res


def is_not_tensor(x):
    return not isinstance(x, torch.Tensor)


def is_float_or_int(x):
    return isinstance(x, (float, int))


def is_numpy_scalar(x):
    return np.isscalar(x)
