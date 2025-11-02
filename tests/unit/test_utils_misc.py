# ruff: noqa: F821, F401
from unittest.mock import MagicMock

import pytest
import torch
from torch.optim import SGD, Adam

from lighter.utils.misc import ensure_list, get_name, get_optimizer_stats, setattr_dot_notation


def test_ensure_list():
    """
    Test the ensure_list function which converts various input types to a list.

    Tests:
        - Converting a single value to a single-item list
        - Preserving an existing list
        - Converting a tuple to a list
    """
    assert ensure_list(1) == [1]
    assert ensure_list([1, 2]) == [1, 2]
    assert ensure_list((1, 2)) == [1, 2]  # Test with tuple input


def test_setattr_dot_notation():
    """
    Test the setattr_dot_notation function which sets attributes using dot notation.

    Tests:
        - Setting a direct attribute on an object
        - Setting a nested attribute using dot notation
        - Verifying that attempting to set a non-existent attribute raises AttributeError

    The function uses dummy classes to simulate nested object structures.
    """

    class Dummy:
        def __init__(self):
            self.attr = MagicMock()

    class NestedDummy:
        def __init__(self):
            self.inner = Dummy()

    obj = Dummy()
    nested_obj = NestedDummy()
    setattr_dot_notation(obj, "attr", 10)
    assert obj.attr == 10

    # Test with nested attribute using dot notation
    setattr_dot_notation(nested_obj, "inner.attr", 20)
    assert nested_obj.inner.attr == 20

    with pytest.raises(AttributeError):
        setattr_dot_notation(obj, "non_existent_attr", 10)


def test_get_name():
    """
    Test the get_name function which retrieves the name of a function or class.

    Tests:
        - Getting the name of a function
        - Getting the name of a class
        - Getting the fully qualified name (including module) of a function

    Verifies both simple name retrieval and module-included name retrieval.
    """

    def sample_function():
        pass

    class Dummy:
        pass

    assert get_name(sample_function) == "sample_function"
    assert get_name(Dummy) == "Dummy"
    assert "test_utils_misc" in get_name(sample_function, include_module_name=True)


def test_get_optimizer_stats():
    """
    Test the get_optimizer_stats function which extracts statistics from PyTorch optimizers.

    Tests:
        - Basic optimizer configuration with single parameter group
        - Complex optimizer configuration with multiple parameter groups

    Verifies:
        - Correct extraction of learning rate and momentum values
        - Proper handling of multiple parameter groups with distinct settings
        - Correct formatting of stat names including group numbers

    Uses SGD optimizer with both single and multiple parameter group configurations
    to ensure comprehensive coverage of optimizer statistics extraction.
    """
    model = torch.nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    stats = get_optimizer_stats(optimizer)
    assert "optimizer/SGD/lr" in stats
    assert stats["optimizer/SGD/lr"] == 0.01
    assert "optimizer/SGD/momentum" in stats
    assert stats["optimizer/SGD/momentum"] == 0.9

    # Test with multiple parameter groups
    # Create separate parameter groups with distinct parameters
    model1 = torch.nn.Linear(10, 1)
    model2 = torch.nn.Linear(10, 1)
    optimizer = SGD(
        [
            {"params": model1.parameters(), "lr": 0.01, "momentum": 0.9},
            {"params": model2.parameters(), "lr": 0.02, "momentum": 0.8},
        ]
    )
    stats = get_optimizer_stats(optimizer)
    assert "optimizer/SGD/lr/group1" in stats
    assert stats["optimizer/SGD/lr/group1"] == 0.01
    assert "optimizer/SGD/momentum/group1" in stats
    assert stats["optimizer/SGD/momentum/group1"] == 0.9
    assert "optimizer/SGD/lr/group2" in stats
    assert stats["optimizer/SGD/lr/group2"] == 0.02
    assert "optimizer/SGD/momentum/group2" in stats
    assert stats["optimizer/SGD/momentum/group2"] == 0.8


def test_get_optimizer_stats_with_betas():
    """
    Test the get_optimizer_stats function with optimizers that use betas instead of momentum.

    Tests:
        - Optimizer with betas parameter (e.g., Adam)
        - Multiple parameter groups with different betas values

    Verifies:
        - Correct extraction of learning rate and beta1 values
        - Proper handling of multiple parameter groups
        - Correct formatting of stat names
    """
    model = torch.nn.Linear(10, 1)
    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    stats = get_optimizer_stats(optimizer)
    assert "optimizer/Adam/lr" in stats
    assert stats["optimizer/Adam/lr"] == 0.001
    assert "optimizer/Adam/momentum" in stats
    assert stats["optimizer/Adam/momentum"] == 0.9  # beta1 value

    # Test with multiple parameter groups with different betas
    model1 = torch.nn.Linear(10, 1)
    model2 = torch.nn.Linear(10, 1)
    optimizer = Adam(
        [
            {"params": model1.parameters(), "lr": 0.001, "betas": (0.9, 0.999)},
            {"params": model2.parameters(), "lr": 0.002, "betas": (0.8, 0.999)},
        ]
    )
    stats = get_optimizer_stats(optimizer)
    assert "optimizer/Adam/lr/group1" in stats
    assert stats["optimizer/Adam/lr/group1"] == 0.001
    assert "optimizer/Adam/momentum/group1" in stats
    assert stats["optimizer/Adam/momentum/group1"] == 0.9
    assert "optimizer/Adam/lr/group2" in stats
    assert stats["optimizer/Adam/lr/group2"] == 0.002
    assert "optimizer/Adam/momentum/group2" in stats
    assert stats["optimizer/Adam/momentum/group2"] == 0.8
