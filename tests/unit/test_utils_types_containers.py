from dataclasses import dataclass, is_dataclass

import pytest
from torchmetrics import Accuracy, MetricCollection


from lighter.utils.types.containers import Metrics


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
