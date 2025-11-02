from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from torchmetrics import Metric, MetricCollection

from lighter.flow import Flow
from lighter.utils.types.enums import Mode


def nested(cls):
    """
    Decorator to handle nested dataclass creation.
    Example:
        ```
        @nested
        @dataclass
        class Example:
            ...
        ```
    """
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        for f in fields(cls):
            if is_dataclass(f.type) and f.name in kwargs:
                kwargs[f.name] = f.type(**kwargs[f.name])
        original_init(self, *args, **kwargs)

    cls.__init__ = __init__
    return cls


@dataclass
class Metrics:
    train: Metric | MetricCollection | None = None
    val: Metric | MetricCollection | None = None
    test: Metric | MetricCollection | None = None

    def __post_init__(self):
        self.train = self._convert_to_collection(self.train)
        self.val = self._convert_to_collection(self.val)
        self.test = self._convert_to_collection(self.test)

    def _convert_to_collection(self, x):
        if x is not None and not isinstance(x, MetricCollection):
            return MetricCollection(x)
        return x


@dataclass
class DataLoaders:
    train: Any | None = None
    val: Any | None = None
    test: Any | None = None
    predict: Any | None = None


@nested
@dataclass
class Flows:
    train: Flow = field(default_factory=lambda: Flow.get_default(Mode.TRAIN))
    val: Flow = field(default_factory=lambda: Flow.get_default(Mode.VAL))
    test: Flow = field(default_factory=lambda: Flow.get_default(Mode.TEST))
    predict: Flow = field(default_factory=lambda: Flow.get_default(Mode.PREDICT))
