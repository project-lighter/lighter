from typing import Any, Callable, Mapping, Sequence

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from monai.transforms import Compose

ROOT_NODE_NAME = "initial"


class ProcessingStep(ABC):

    @property
    @abstractmethod
    def is_constant(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, data: Any):
        raise NotImplementedError()

    @property
    @abstractmethod
    def requires(self) -> Sequence[str]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def should_cache_result(self) -> bool:
        raise NotImplementedError()


class TransformConversionDescriptor:

    @staticmethod
    def _convert_to_transform(value) -> Callable:

        if isinstance(value, Callable):
            return value
        elif isinstance(value, Sequence):
            return Compose(transforms=value)
        else:
            raise ValueError("Value Callable or Sequence of Callables")

    def __init__(self, *, default=None):
        if default is None:
            self._default = None
        else:
            self._default = self._convert_to_transform(default)

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, type) -> Callable:
        if obj is None:
            return self._default
        if self._default is None:
            return getattr(obj, self._name)
        else:
            return getattr(obj, self._name, self._default)

    def __set__(self, obj, value):
        setattr(obj, self._name, self._convert_to_transform(value))


@dataclass(slots=True, frozen=True)
class TransformProcessingStep(ProcessingStep):

    transform: TransformConversionDescriptor = TransformConversionDescriptor()
    requires: Sequence[str] = (ROOT_NODE_NAME,)
    should_cache_result: bool = True

    @property
    def is_constant(self) -> bool:
        return False

    def __call__(self, data: Any):
        return self.transform(data)


class ConstantProcessingStep(ProcessingStep):

    @property
    def should_cache_result(self) -> bool:
        return True

    @property
    def is_constant(self) -> bool:
        return True

    def __call__(self, data: Any):
        raise RuntimeError("Cannot call a constant processing step")

    @property
    def requires(self) -> Sequence[str]:
        return []


class ProcessingPipelineDefinition(Mapping[str, ProcessingStep]):

    processing_steps: Mapping[str, ProcessingStep]

    def __init__(self, processing_steps: Mapping[str, ProcessingStep | dict], add_initial_node: bool = True):
        processing_steps = {k: self._parse_step(v) for k, v in processing_steps.items()}
        if add_initial_node:
            processing_steps[ROOT_NODE_NAME] = ConstantProcessingStep()
        self.processing_steps = processing_steps
        self._check_graph()

    @staticmethod
    def _parse_step(step: Any):
        if isinstance(step, ProcessingStep):
            return step
        elif isinstance(step, dict):
            return TransformProcessingStep(**step)
        else:
            raise ValueError("Step must be a ProcessingStep or a dict")

    def _check_graph(self):
        for step_name, step in self.processing_steps.items():
            for required_step in step.requires:
                if required_step not in self.processing_steps:
                    raise ValueError(f"Step {step_name} requires step {required_step} which is not defined")
        self._check_graph_for_cycles()

    # Check for any cycles in the graph
    # TODO: This is not very efficient, but for the size of graphs that are expected, this shouldn't matter
    def _check_graph_for_cycles(self):
        def _dfs(node_name: str, visited: set):
            if node_name in visited:
                raise ValueError("Cyclic dependency detected")
            visited.add(node_name)
            for required_step in self.processing_steps[node_name].requires:
                _dfs(required_step, visited)
            visited.remove(node_name)

        # We need to check every node, since the nodes might not all be connected
        for node in self.processing_steps.keys():
            _dfs(node, set())

    def __getitem__(self, __key):
        return self.processing_steps[__key]

    def __iter__(self):
        return iter(self.processing_steps)

    def __len__(self):
        return len(self.processing_steps)


class ProcessingPipeline:
    pipeline: ProcessingPipelineDefinition
    results: dict[str, Any]

    def __init__(self, pipeline: ProcessingPipelineDefinition, start_value: Any, start_node: str = ROOT_NODE_NAME):
        self.pipeline = pipeline
        self.results = {start_node: start_value}

    def get_result(self, node_name: str):
        if node_name not in self.pipeline:
            raise ValueError(
                f"Step {node_name} not found in processing pipeline, defined step are {list(self.pipeline.keys())}"
            )
        if node_name in self.results:
            return self.results[node_name]
        step = self.pipeline[node_name]
        if step.is_constant:
            raise ValueError(f"Step {node_name} is constant and has not been assigned a value")
        errors = []
        # data = None
        for required_step in step.requires:
            try:
                data = self.get_result(required_step)
                break
            except Exception as e:
                errors.append(e)
        else:
            raise ExceptionGroup(f"Could not find any valid required data for step {node_name}", errors)

        result = step(data)
        if step.should_cache_result:
            self.results[node_name] = result
        return result
