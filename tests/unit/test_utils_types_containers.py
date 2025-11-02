from dataclasses import dataclass, is_dataclass

import pytest
from torchmetrics import Accuracy, MetricCollection

from lighter.flow import Flow
from lighter.utils.types.containers import Flows, Metrics, nested


# Define a nested dataclass for testing
@nested
@dataclass
class Inner:
    value: int


@nested
@dataclass
class Outer:
    inner: Inner
    name: str = "default"


def test_nested_decorator_initialization():
    """Test that the nested decorator correctly initializes nested dataclasses."""
    data = {"inner": {"value": 10}, "name": "test"}
    outer_instance = Outer(**data)

    assert isinstance(outer_instance, Outer)
    assert isinstance(outer_instance.inner, Inner)
    assert outer_instance.inner.value == 10
    assert outer_instance.name == "test"


def test_nested_decorator_default_values():
    """Test that the nested decorator respects default values."""
    data = {"inner": {"value": 5}}
    outer_instance = Outer(**data)

    assert outer_instance.name == "default"


def test_nested_decorator_is_dataclass():
    """Test that the decorated class is still recognized as a dataclass."""
    assert is_dataclass(Outer)
    assert is_dataclass(Inner)


def test_nested_decorator_missing_nested_data():
    """Test that missing nested data raises a TypeError."""
    with pytest.raises(TypeError):
        Outer(name="test")


def test_metrics_convert_to_collection():
    """Test that _convert_to_collection converts non-MetricCollection to MetricCollection."""

    # Create a Metrics instance with a single Metric
    accuracy_metric = Accuracy(task="binary")
    metrics_instance = Metrics(train=accuracy_metric)

    # Check if the train metric is converted to a MetricCollection
    assert not isinstance(accuracy_metric, MetricCollection)
    assert isinstance(metrics_instance.train, MetricCollection)
    assert accuracy_metric in metrics_instance.train.values()


def test_metrics_convert_none_to_collection():
    """Test that _convert_to_collection handles None values correctly."""
    metrics_instance = Metrics(train=None, val=None, test=None)

    assert metrics_instance.train is None
    assert metrics_instance.val is None
    assert metrics_instance.test is None


def test_flows_default_factory():
    """Test that the Flows dataclass uses the correct default factories."""
    flows_instance = Flows()

    assert isinstance(flows_instance.train, Flow)
    assert isinstance(flows_instance.val, Flow)
    assert isinstance(flows_instance.test, Flow)
    assert isinstance(flows_instance.predict, Flow)
