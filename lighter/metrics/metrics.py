from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from functools import singledispatch
from typing import Callable, Sequence, Protocol, Mapping, Any
from copy import deepcopy

import torch
from monai.utils import convert_to_tensor

from lighter.postprocessing.pipeline import ProcessingPipeline
from lighter.defs import ModeEnum


class MetricAdapter(Protocol):

    def __call__(self, metric: Callable, batch: Mapping):
        return NotImplemented

class DefaultMetricAdapter(MetricAdapter):

    def __call__(self, metric: Callable, batch: Mapping):
        return metric(batch["pred"], batch["target"])

MetricResult = Any

@singledispatch
def reset_metric(metric: Any, container: MetricContainer):
    return NotImplemented

@singledispatch
def aggregate_metric(metric: Any, container: MetricContainer):
    return NotImplemented

@singledispatch
def get_metric_name(metric: Any):
    if hasattr(metric, "__name__"):
        return metric.__name__
    if hasattr(metric, "__class__"):
        return metric.__class__.__name__
    return NotImplemented

class MetricResultDims(StrEnum):
    BATCH = auto()
    CLASS = auto()

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None

@dataclass(init=False, eq=False, slots=True)
class MetricContainer:
    metric: Callable
    _metrics: dict[str, Callable] = field(repr=False)
    adapter: MetricAdapter = field(repr=False)
    modes: Sequence[str]
    processing_step: str | None
    step_dims: Sequence[MetricResultDims]
    compute_dims: Sequence[MetricResultDims]
    class_names: Sequence[str]
    name: str
    def __init__(self, metric: Callable, adapter: MetricAdapter = DefaultMetricAdapter(),
                 modes: Sequence[str] = (ModeEnum.VAL, ModeEnum.TEST),
                 processing_step: str | None = None, name: str | None = None,
                 step_dims: Sequence[str] = tuple(),
                 compute_dims: Sequence[str] = tuple(),
                 class_names: Sequence[str] = tuple()
                 ):
        self.metric = metric
        self._metrics = dict()
        self.adapter = adapter
        self.modes = modes
        self.processing_step = processing_step
        self.name = name
        if name is None:
            self.name = get_metric_name(metric)
        self.step_dims = self._clean_dims_attribue(step_dims)
        self.compute_dims = self._clean_dims_attribue(compute_dims)
        self.class_names = class_names

    def _clean_dims_attribue(self, dims: Sequence[str]) -> Sequence[MetricResultDims]:
        result = []
        for dim in dims:
            dim = MetricResultDims(dim)
            if dim is None:
                raise ValueError(f"Invalid dimension {dim}, "
                                 f"must be one of {MetricResultDims.__members__}")
            result.append(dim)
        if len(result) > len(set(result)):
            raise ValueError(f"Duplicate dimensions in metric {self.name}")
        return result
    def _get_metric_instance(self, mode: ModeEnum, create_if_not_exists: bool = False):
        if mode not in self.modes:
            return None
        if mode not in self._metrics:
            if not create_if_not_exists:
                return None
            self._metrics[mode] = deepcopy(self.metric)
        return self._metrics[mode]

    def _check_dims(self, result: torch.Tensor, dims:Sequence[MetricResultDims]):
        if result.dim() != len(dims):
            if len(dims) == 0 and result.numel() == 1:
                # Some metrics return an n-dimensional tensor with a single element, which we will
                # treat as a scalar. This might lead this check to miss cases where there is a batch dimension,
                # but the metric was only called on a single sample.
                return
            raise ValueError(f"Invalid number of dimensions in result,"
                             f" expected {len(dims)} ({dims}), got {result.dim()}. "
                             "Please check the step_dims and compute_dims attributes of the metric."
                             )
        try:
            class_dim_index = dims.index(MetricResultDims.CLASS)
            if result.size(class_dim_index) != len(self.class_names):
                raise ValueError(f"Invalid number of classes in result, expected {len(self.class_names)} ({self.class_names}), got {result.size(class_dim_index)}")
        except ValueError:
            pass
    def __call__(self, data: dict, mode: ModeEnum) -> MetricResult | None:
        metric = self._get_metric_instance(mode, create_if_not_exists=True)
        if metric is None:
            return None
        if self.processing_step is not None:
            pipeline: ProcessingPipeline = data["pipeline"]
            data = pipeline.get_result(self.processing_step)
        result = convert_to_tensor(self.adapter(metric, data), track_meta=True)
        self._check_dims(result, self.step_dims)
        return result

    def reset(self, mode: ModeEnum):
        if metric := self._get_metric_instance(mode):
            reset_metric(metric, self)

    def compute(self, mode: ModeEnum)-> MetricResult | None:
        if metric := self._get_metric_instance(mode):
            result = convert_to_tensor(aggregate_metric(metric, self), track_meta=True)
            self._check_dims(result, self.compute_dims)
            return result
        return None



class MetricContainerCollection:
    metrics: list[MetricContainer]

    def __init__(self, metrics: list[MetricContainer]):
        self.metrics = metrics

    def __call__(self, batch: dict, mode: ModeEnum) -> dict[MetricContainer,MetricResult]:
        results = dict()
        for metric in self.metrics:
            try:
                result = metric(batch, mode)
            except Exception as e:
                raise RuntimeError(f"Error computing metric {metric}") from e
            if result is not None:
                results[metric] = result
        return results

    def reset(self, mode: ModeEnum):
        for metric in self.metrics:
            try:
                metric.reset(mode)
            except Exception as e:
                raise RuntimeError(f"Error resetting metric {metric}") from e

    def compute(self, mode: ModeEnum) -> dict[MetricContainer,MetricResult]:
        results = dict()
        for metric in self.metrics:
            try:
                result = metric.compute(mode)
            except Exception as e:
                raise RuntimeError(f"Error computing metric {metric}") from e
            if result is not None:
                results[metric] = result
        return results


# Import the metric integrations
import lighter.metrics.integration # noqa
