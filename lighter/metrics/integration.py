from __future__ import annotations

from typing import TYPE_CHECKING, Any

from functools import singledispatch

import monai.metrics
import torchmetrics

from lighter.metrics.metrics import MetricContainer, aggregate_metric, reset_metric


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
