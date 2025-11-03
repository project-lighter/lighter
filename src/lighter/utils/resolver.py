import re
from collections.abc import Callable
from typing import Any

from torch import Tensor


def resolve_value(data: dict[str, Any], key: str | Callable[..., Any] | list[Any]) -> Any:
    """
    Resolves a value from `data` using a flexible `key`.

    The `key` can be:
    - `str`: A string for direct, nested (e.g., 'a.b'), indexed (e.g., 'a[0]'),
             or special 'batch' access (e.g., 'batch[0]', 'batch["image"]').
    - `Callable`: A function `f(data)` that returns the desired value.
    - `list[Any]`: A pipeline of keys/callables applied sequentially.

    Example:
        >>> data = {'a': {'b': [10, 20]}, 'c': 30, 'batch': [1, 2]}
        >>> resolve_value(data, 'a.b[1]')
        20
        >>> resolve_value(data, lambda d: d['c'] * 2)
        60
        >>> resolve_value(data, ['batch[0]', lambda x: x * 10])
        10

    Args:
        data: The dictionary to extract the value from.
        key: The key, callable, or pipeline to use for resolution.

    Returns:
        The resolved value.

    Raises:
        TypeError: If `key` type is unsupported.
    """
    if isinstance(key, list):
        return resolve_pipeline(data, key)
    if callable(key):
        return key(data)
    if isinstance(key, str):
        return resolve_string_key(data, key)
    raise TypeError(f"Unsupported key type: {type(key)}")


def resolve_pipeline(data: dict[str, Any], pipeline: list[Any]) -> Any:
    """
    Applies a pipeline of keys/callables to resolve a value.

    The first element of `pipeline` is resolved from `data`. Subsequent elements
    (which must be callables) transform the result of the previous step.

    Example:
        >>> data = {'features': [1, 2, 3]}
        >>> resolve_pipeline(data, ['features[0]', lambda x: x * 2])
        2

    Args:
        data: The dictionary containing the initial data.
        pipeline: A list of keys/callables.

    Returns:
        The final transformed value.
    """
    value = resolve_value(data, pipeline[0])
    for transform in pipeline[1:]:
        value = transform(value)
    return value


def resolve_string_key(data: dict[str, Any], key: str) -> Any:
    """
    Resolves a value from `data` using a string `key` with various formats.

    Prioritizes:
    1. 'batch' bracket notation (e.g., 'batch[0]').
    2. General indexed access (e.g., 'pred[0]').
    3. Nested dot notation (e.g., 'loss.total').
    4. Direct dictionary key lookup.
    5. Object attribute lookup.

    Example:
        >>> data = {'loss': {'total': 0.5}, 'pred': [1, 2], 'batch': {'id': 1}}
        >>> resolve_string_key(data, 'loss.total')
        0.5
        >>> resolve_string_key(data, 'pred[0]')
        1
        >>> resolve_string_key(data, 'batch["id"]')
        1

    Args:
        data: The dictionary to extract the value from.
        key: The string key to resolve.

    Returns:
        The resolved value.

    Raises:
        KeyError: If the key cannot be resolved.
    """
    if key.startswith("batch["):
        return resolve_batch_bracket_access(data, key)

    # Check for general indexing (e.g., 'pred[0]')
    if '[' in key and key.endswith(']'):
        return resolve_indexed_access(data, key)

    # Check for nested key resolution (e.g., 'loss.total')
    if "." in key:
        return resolve_nested_dot_access(data, key)

    # Direct key lookup
    if key in data:
        return data[key]

    # Fallback to attribute lookup
    if hasattr(data, key):
        return getattr(data, key)

    raise KeyError(f"Could not resolve key '{key}' from data.")


def resolve_batch_bracket_access(data: dict[str, Any], key: str) -> Any:
    """
    Handles 'batch' bracket notation (e.g., 'batch[0]', 'batch["image"]').

    Extracts integer indices or string keys from within the brackets to access
    elements within the 'batch' entry of the `data` dictionary.

    Example:
        >>> data = {'batch': [10, 20]}
        >>> resolve_batch_bracket_access(data, 'batch[0]')
        10
        >>> data = {'batch': {'image': 'img.png'}}
        >>> resolve_batch_bracket_access(data, 'batch["image"]')
        'img.png'

    Args:
        data: Dictionary containing the 'batch' entry.
        key: String key in "batch[..." format.

    Returns:
        The value accessed from the 'batch' element.

    Raises:
        KeyError: If 'batch' not found or nested resolution fails.
        ValueError: If an unsupported accessor format is used.
    """
    try:
        value = data["batch"]
        accessors = re.findall(r"\[(.*?)\]", key)
        for acc in accessors:
            acc = acc.strip()
            if (acc.startswith('"') and acc.endswith('"')) or (acc.startswith("'") and acc.endswith("'")):
                value = value[acc[1:-1]]
            elif acc.isdigit() or (acc.startswith("-") and acc[1:].isdigit()):
                value = value[int(acc)]
            else:
                raise ValueError(f"Unsupported accessor '{acc}' in key '{key}'")
        return value
    except (KeyError, IndexError, ValueError) as e:
        raise KeyError(f"Could not resolve nested key '{key}' from data.\n{e}") from e


def resolve_indexed_access(data: dict[str, Any], key: str) -> Any:
    """
    Handles indexed access for lists, tuples, and `torch.Tensor` (e.g., 'pred[0]').

    Extracts the base key and integer index, then applies the index to the resolved
    base value.

    Example:
        >>> data = {'preds': [100, 200]}
        >>> resolve_indexed_access(data, 'preds[1]')
        200

    Args:
        data: The dictionary to extract the value from.
        key: String key in "base_key[index]" format.

    Returns:
        The indexed value.

    Raises:
        TypeError: If base value is not indexable.
        ValueError: If index is not a valid integer.
    """
    base_key, index_str = key[:-1].split('[')
    index = int(index_str)
    value = resolve_value(data, base_key)
    if isinstance(value, (list, tuple, Tensor)):
        return value[index]
    else:
        raise TypeError(f"Cannot index type {type(value).__name__} with key '{key}'")


def resolve_nested_dot_access(data: dict[str, Any], key: str) -> Any:
    """
    Handles nested key resolution using dot notation (e.g., 'loss.total').

    Iteratively accesses elements within dictionaries, lists/tuples (with integer
    indices), or attributes of objects.

    Example:
        >>> class MyObject:
        ...     def __init__(self):
        ...         self.metrics = {'accuracy': 0.9}
        >>> data = {'results': MyObject()}
        >>> resolve_nested_dot_access(data, 'results.metrics.accuracy')
        0.9

    Args:
        data: The dictionary to extract the value from.
        key: String key in "level1.level2.level3" format.

    Returns:
        The value resolved from the nested structure.

    Raises:
        KeyError: If a key/attribute in the path is not found.
        TypeError: If non-integer key on list/tuple.
    """
    value = data
    for k in key.split("."):
        if isinstance(value, dict):
            if k in value:
                value = value[k]
            else:
                raise KeyError(f"Could not resolve nested key '{key}' from data. "
                               f"Dictionary does not contain key '{k}'")
        elif isinstance(value, (list, tuple)):
            if k.isdigit():
                value = value[int(k)]
            else:
                raise KeyError(f"Cannot access non-integer key '{k}' on list/tuple in nested key '{key}'")
        elif hasattr(value, k):
            value = getattr(value, k)
        else:
            raise KeyError(f"Could not resolve nested key '{key}' from data. "
                           f"'{type(value).__name__}' object has no attribute or key '{k}'")
    return value
