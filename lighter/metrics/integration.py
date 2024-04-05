from __future__ import annotations
from functools import singledispatch
from typing import Any
from typing import TYPE_CHECKING
import monai.metrics
import torchmetrics

from lighter.metrics.metrics import MetricContainer, reset_metric, aggregate_metric


@reset_metric.register
def _(metric: monai.metrics.Cumulative, container: MetricContainer):
    metric.reset()

@reset_metric.register
def _(metric: torchmetrics.Metric, container: MetricContainer):
    metric.reset()



@aggregate_metric.register
def _(metric: monai.metrics.Cumulative, container: MetricContainer):
    return metric.aggregate()

@aggregate_metric.register
def _(metric: torchmetrics.Metric, container: MetricContainer):
    return metric.compute()
