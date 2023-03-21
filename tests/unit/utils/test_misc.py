from types import SimpleNamespace

import pytest

from lighter.system import LighterSystem
from lighter.utils.misc import countargs, ensure_list, get_name, hasarg, setattr_dot_notation


def test_ensure_list():
    """Test the ensure_list function."""
    # Test with list
    assert ensure_list([1, 2, 3]) == [1, 2, 3]

    # Test with tuple
    assert ensure_list((1, 2, 3)) == [1, 2, 3]

    # Test with string
    assert ensure_list("hello") == ["hello"]

    # Test with integer
    assert ensure_list(42) == [42]

    # Test with float
    assert ensure_list(3.14) == [3.14]

    # Test with boolean
    assert ensure_list(True) == [True]

    # Test with None
    assert ensure_list(None) == []

    # Test with empty list
    assert ensure_list([]) == []

    # Test with empty tuple
    assert ensure_list(()) == []

    # Test with nested list
    assert ensure_list([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]


def test_setattr_dot_notation():
    """Test the setattr_dot_notation function."""

    class TestObject:
        def __init__(self):
            self.foo = "bar"
            self.baz = SimpleNamespace(qux="quux")

    obj = TestObject()

    # Test setting attribute with single name
    setattr_dot_notation(obj, "foo", "new_value")
    assert obj.foo == "new_value"

    # Test setting attribute with dot notation
    setattr_dot_notation(obj, "baz.qux", "new_value")
    assert obj.baz.qux == "new_value"

    # Test setting non-existent attribute
    with pytest.raises(AttributeError):
        setattr_dot_notation(obj, "non_existent_attr", "value")

    # Test setting non-existent attribute with dot notation
    with pytest.raises(AttributeError):
        setattr_dot_notation(obj, "non_existent_attr.attr2", "value")


def test_hasarg():
    """Test the hasarg function. Tests with a function, a class and a class method."""

    def my_func(arg1, arg2, *, arg3=None):
        pass

    class MyClass:
        def __init__(self, arg1, arg2):
            pass

        def my_method(self, arg1, arg2, *, arg3=None):
            pass

    # Test function with positional and keyword arguments
    assert hasarg(my_func, "arg1")
    assert hasarg(my_func, "arg2")
    assert hasarg(my_func, "arg3")

    # Test function with only keyword argument
    assert not hasarg(my_func, "arg4")

    # Test class constructor with positional arguments
    assert hasarg(MyClass.__init__, "arg1")
    assert hasarg(MyClass.__init__, "arg2")

    # Test class method with positional and keyword arguments
    assert hasarg(MyClass.my_method, "arg1")
    assert hasarg(MyClass.my_method, "arg2")
    assert hasarg(MyClass.my_method, "arg3")

    # Test class method with only keyword argument
    assert not hasarg(MyClass.my_method, "arg4")


def test_countargs():
    """Test the countargs function.
    Tests with a function, a class and a class method,
    an object's method,and a lambda function.
    """

    # Test with a function
    def my_function(x, y, z=3):
        pass

    assert countargs(my_function) == 3

    # Test with a class and a class method
    class MyClass:
        def __init__(self, a, b, c, d) -> None:
            pass

        def my_method(self, a, b):
            pass

    assert countargs(MyClass) == 4
    assert countargs(MyClass.my_method) == 2

    # Test with an object's method
    my_class = MyClass(1, 2, 3, 4)
    assert countargs(my_class.my_method) == 2

    # Test with a lambda function
    my_lambda = lambda x, y: x**y
    assert countargs(my_lambda) == 2


def test_get_name():
    """Test the get_name function.
    Tests with a function, a class and an object.
    """
    # Test with a function
    assert get_name(get_name) == "get_name"
    assert get_name(get_name, include_module_name=True) == "lighter.utils.misc.get_name"

    # Test with a class
    assert get_name(LighterSystem) == "LighterSystem"
    assert get_name(LighterSystem, include_module_name=True) == "lighter.system.LighterSystem"

    # Test with an object
    system = LighterSystem(None, 1)
    assert get_name(system) == "LighterSystem"
    assert get_name(system, include_module_name=True) == "lighter.system.LighterSystem"
