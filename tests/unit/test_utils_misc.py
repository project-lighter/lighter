"""Unit tests for utility functions in lighter/utils/misc.py"""

import torch
from torch.optim import SGD, Adam

from lighter.utils.misc import ensure_list, get_name, get_optimizer_stats, hasarg


def test_ensure_list_with_list():
    """Test ensure_list returns list as-is."""
    input_list = [1, 2, 3]
    assert ensure_list(input_list) == [1, 2, 3]
    assert ensure_list(input_list) is input_list  # Should return same object


def test_ensure_list_with_tuple():
    """Test ensure_list converts tuple to list."""
    assert ensure_list((1, 2, 3)) == [1, 2, 3]


def test_ensure_list_with_none():
    """Test ensure_list returns empty list for None."""
    assert ensure_list(None) == []


def test_ensure_list_with_single_value():
    """Test ensure_list wraps single value."""
    assert ensure_list(42) == [42]
    assert ensure_list("string") == ["string"]


def test_hasarg_with_function():
    """Test hasarg with a simple function."""

    def test_func(a, b, c=10):
        return a + b + c

    assert hasarg(test_func, "a") is True
    assert hasarg(test_func, "b") is True
    assert hasarg(test_func, "c") is True
    assert hasarg(test_func, "d") is False


def test_hasarg_with_method():
    """Test hasarg with a class method."""

    class TestClass:
        def method(self, x, y):
            return x + y

    assert hasarg(TestClass.method, "self") is True
    assert hasarg(TestClass.method, "x") is True
    assert hasarg(TestClass.method, "y") is True
    assert hasarg(TestClass.method, "z") is False


def test_get_name_without_module():
    """Test get_name without module name."""

    def test_function():
        pass

    class TestClass:
        pass

    assert get_name(test_function) == "test_function"
    assert get_name(TestClass) == "TestClass"


def test_get_name_with_module():
    """Test get_name with module name."""

    def test_function():
        pass

    # The test function's module is __main__ during testing
    name = get_name(test_function, include_module_name=True)
    assert "test_function" in name


def test_get_optimizer_stats_single_group():
    """Test get_optimizer_stats with single parameter group."""
    model = torch.nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    stats = get_optimizer_stats(optimizer)

    assert "optimizer/SGD/lr" in stats
    assert "optimizer/SGD/momentum" in stats
    assert stats["optimizer/SGD/lr"] == 0.01
    assert stats["optimizer/SGD/momentum"] == 0.9


def test_get_optimizer_stats_multiple_groups():
    """Test get_optimizer_stats with multiple parameter groups."""
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
    assert "optimizer/SGD/lr/group2" in stats
    assert "optimizer/SGD/momentum/group1" in stats
    assert "optimizer/SGD/momentum/group2" in stats
    assert stats["optimizer/SGD/lr/group1"] == 0.01
    assert stats["optimizer/SGD/lr/group2"] == 0.02
    assert stats["optimizer/SGD/momentum/group1"] == 0.9
    assert stats["optimizer/SGD/momentum/group2"] == 0.8


def test_get_optimizer_stats_with_betas():
    """Test get_optimizer_stats with Adam optimizer (uses betas instead of momentum)."""
    model = torch.nn.Linear(10, 1)
    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    stats = get_optimizer_stats(optimizer)
    assert "optimizer/Adam/lr" in stats
    assert stats["optimizer/Adam/lr"] == 0.001
    # Adam reports beta1 and beta2, not momentum
    assert "optimizer/Adam/beta1" in stats
    assert "optimizer/Adam/beta2" in stats
    assert stats["optimizer/Adam/beta1"] == 0.9
    assert stats["optimizer/Adam/beta2"] == 0.999

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
    assert "optimizer/Adam/lr/group2" in stats
    assert "optimizer/Adam/beta1/group1" in stats
    assert "optimizer/Adam/beta1/group2" in stats
    assert stats["optimizer/Adam/lr/group1"] == 0.001
    assert stats["optimizer/Adam/lr/group2"] == 0.002
    assert stats["optimizer/Adam/beta1/group1"] == 0.9
    assert stats["optimizer/Adam/beta1/group2"] == 0.8


def test_get_optimizer_stats_no_momentum():
    """Test get_optimizer_stats with optimizer without momentum."""
    model = torch.nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0)  # No momentum

    stats = get_optimizer_stats(optimizer)

    assert "optimizer/SGD/lr" in stats
    assert stats["optimizer/SGD/lr"] == 0.01
    # Should still include momentum even if it's 0
    assert "optimizer/SGD/momentum" in stats
    assert stats["optimizer/SGD/momentum"] == 0


def test_get_optimizer_stats_with_weight_decay():
    """Test get_optimizer_stats includes weight decay when non-zero."""
    model = torch.nn.Linear(10, 1)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    stats = get_optimizer_stats(optimizer)

    assert "optimizer/Adam/weight_decay" in stats
    assert stats["optimizer/Adam/weight_decay"] == 0.01


def test_get_optimizer_stats_zero_weight_decay():
    """Test get_optimizer_stats excludes weight decay when zero."""
    model = torch.nn.Linear(10, 1)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0)

    stats = get_optimizer_stats(optimizer)

    # Weight decay should not be in stats when it's 0
    assert "optimizer/Adam/weight_decay" not in stats
