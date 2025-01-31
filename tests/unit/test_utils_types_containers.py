from dataclasses import dataclass, is_dataclass

import pytest
from torchmetrics import Accuracy, MetricCollection

from lighter.adapters import BatchAdapter, CriterionAdapter, LoggingAdapter, MetricsAdapter
from lighter.utils.types.containers import Metrics, Predict, Test, Train, Val, nested


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


def test_train_default_factory():
    """Test that the Train dataclass uses the correct default factories."""
    train_instance = Train()

    assert isinstance(train_instance.batch, BatchAdapter)
    assert train_instance.batch.input_accessor == 0
    assert train_instance.batch.target_accessor == 1

    assert isinstance(train_instance.criterion, CriterionAdapter)
    assert train_instance.criterion.pred_argument == 0
    assert train_instance.criterion.target_argument == 1

    assert isinstance(train_instance.metrics, MetricsAdapter)
    assert train_instance.metrics.pred_argument == 0
    assert train_instance.metrics.target_argument == 1

    assert isinstance(train_instance.logging, LoggingAdapter)


def test_val_default_factory():
    """Test that the Val dataclass uses the correct default factories."""
    val_instance = Val()

    assert isinstance(val_instance.batch, BatchAdapter)
    assert val_instance.batch.input_accessor == 0
    assert val_instance.batch.target_accessor == 1

    assert isinstance(val_instance.criterion, CriterionAdapter)
    assert val_instance.criterion.pred_argument == 0
    assert val_instance.criterion.target_argument == 1

    assert isinstance(val_instance.metrics, MetricsAdapter)
    assert val_instance.metrics.pred_argument == 0
    assert val_instance.metrics.target_argument == 1

    assert isinstance(val_instance.logging, LoggingAdapter)


def test_test_default_factory():
    """Test that the Test dataclass uses the correct default factories."""
    test_instance = Test()

    assert isinstance(test_instance.batch, BatchAdapter)
    assert test_instance.batch.input_accessor == 0
    assert test_instance.batch.target_accessor == 1

    assert isinstance(test_instance.metrics, MetricsAdapter)
    assert test_instance.metrics.pred_argument == 0
    assert test_instance.metrics.target_argument == 1

    assert isinstance(test_instance.logging, LoggingAdapter)


def test_predict_default_factory():
    """Test that the Predict dataclass uses the correct default factories."""
    predict_instance = Predict()

    assert isinstance(predict_instance.batch, BatchAdapter)
    assert callable(predict_instance.batch.input_accessor)
    assert predict_instance.batch.input_accessor("this is a batch") == "this is a batch"

    assert isinstance(predict_instance.logging, LoggingAdapter)
