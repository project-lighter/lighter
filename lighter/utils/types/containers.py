from typing import Any, Callable

from dataclasses import dataclass, field

from torch import nn
from torchmetrics import Metric, MetricCollection

from lighter.adapters import BatchAdapter, CriterionAdapter, LoggingAdapter, MetricsAdapter


class Metrics(nn.Module):
    def __init__(self, train=None, val=None, test=None):
        super().__init__()
        self.train = MetricCollection(train) if train is not None else {}
        self.val = MetricCollection(val) if val is not None else {}
        self.test = MetricCollection(test) if test is not None else {}


@dataclass
class DataLoaders:
    train: Any | None = None
    val: Any | None = None
    test: Any | None = None
    predict: Any | None = None


@dataclass
class Train:
    """Train mode sub-dataclass for Adapters."""

    batch: BatchAdapter = field(default_factory=lambda: BatchAdapter(input_accessor="input", target_accessor="target"))
    criterion: CriterionAdapter = field(default_factory=lambda: CriterionAdapter(pred_argument=0, target_argument=1))
    metrics: MetricsAdapter = field(default_factory=lambda: MetricsAdapter(pred_argument=0, target_argument=1))
    logging: LoggingAdapter = field(default_factory=LoggingAdapter)


@dataclass
class Val:
    """Val mode sub-dataclass for Adapters."""

    batch: BatchAdapter = field(default_factory=lambda: BatchAdapter(input_accessor="input", target_accessor="target"))
    criterion: CriterionAdapter = field(default_factory=lambda: CriterionAdapter(pred_argument=0, target_argument=1))
    metrics: MetricsAdapter = field(default_factory=lambda: MetricsAdapter(pred_argument=0, target_argument=1))
    logging: LoggingAdapter = field(default_factory=LoggingAdapter)


@dataclass
class Test:
    """Test mode sub-dataclass for Adapters."""

    batch: BatchAdapter = field(default_factory=lambda: BatchAdapter(input_accessor="input", target_accessor="target"))
    metrics: MetricsAdapter = field(default_factory=lambda: MetricsAdapter(pred_argument=0, target_argument=1))
    logging: LoggingAdapter = field(default_factory=LoggingAdapter)


@dataclass
class Predict:
    """Predict mode sub-dataclass for Adapters."""

    batch: BatchAdapter = field(default_factory=lambda: BatchAdapter(input_accessor="input"))
    logging: LoggingAdapter = field(default_factory=LoggingAdapter)


@dataclass
class Adapters:
    """Root configuration class for all adapters across different modes."""

    train: Train = field(default_factory=Train)
    val: Val = field(default_factory=Val)
    test: Test = field(default_factory=Test)
    predict: Predict = field(default_factory=Predict)

    def __post_init__(self):
        """Ensure nested dataclasses are properly initialized."""
        if isinstance(self.train, dict):
            self.train = Train(**self.train)
        if isinstance(self.val, dict):
            self.val = Val(**self.val)
        if isinstance(self.test, dict):
            self.test = Test(**self.test)
        if isinstance(self.predict, dict):
            self.predict = Predict(**self.predict)
