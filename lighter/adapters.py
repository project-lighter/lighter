from typing import Any, Callable

from abc import ABC

from lighter.utils.misc import ensure_list


class TransformAdapter(ABC):
    """
    An abstract base class for applying transform functions to data.
    """

    def _transform(self, data: Any, transforms: Callable | list[Callable]) -> Any:
        """
        Applies a list of transform functions to the data.

        Args:
            data: The data to be transformed.
            transforms: A single transform function or a list of functions.

        Returns:
            The transformed data.

        Raises:
            ValueError: If any transform is not callable.
        """
        for transform in ensure_list(transforms):
            if callable(transform):
                data = transform(data)
            else:
                raise ValueError(f"Invalid transform type for transform: {transform}")
        return data


class BatchAdapter:
    def __init__(
        self,
        input_accessor: int | str | Callable | None = None,
        target_accessor: int | str | Callable | None = None,
        identifier_accessor: int | str | Callable | None = None,
    ):
        """
        Initializes BatchAdapter with accessors for input, target, and id.

        Args:
            input_accessor: Accessor for the input data. Can be an index (for lists/tuples),
                            a key (for dictionaries), or a callable.
            target_accessor: Accessor for the target data. Can be an index (for lists/tuples),
                            a key (for dictionaries), or a callable.
            identifier_accessor: Accessor for the identifier data. Can be an index (for lists/tuples),
                            a key (for dictionaries), or a callable.
        """
        self.input_accessor = input_accessor
        self.target_accessor = target_accessor
        self.identifier_accessor = identifier_accessor

    def identifier(self, data: Any) -> Any:
        # TODO - see what to do regarding the default value, old lighter would return None if id doesnt exist
        return self._access_value(data, self.identifier_accessor)

    def input(self, data: Any) -> Any:
        return self._access_value(data, self.input_accessor)

    def target(self, data: Any) -> Any:
        return self._access_value(data, self.target_accessor)

    def _access_value(self, data: Any, accessor: int | str | Callable) -> Any:
        """
        Accesses a value from the data using the provided accessor.

        Args:
            data: The data to access the value from.
            accessor: The accessor to use. Can be an index (for lists/tuples),
                      a key (for dictionaries), or a callable.

        Returns:
            The accessed value.

        Raises:
            ValueError: If the accessor type or data structure is invalid.
        """
        if accessor is None:
            return data
        elif isinstance(accessor, int) and isinstance(data, (tuple, list)):
            return data[accessor]
        elif isinstance(accessor, str) and isinstance(data, dict):
            return data.get(accessor)
        elif callable(accessor):
            return accessor(data)
        else:
            raise ValueError(f"Invalid accessor {accessor} of type {type(accessor)} for data type {type(data)}.")


class FunctionAdapter(TransformAdapter):
    """
    A generic adapter for applying functions (criterion or metrics) to data.
    """

    def __init__(
        self,
        input_argument: int | str | None = None,
        target_argument: int | str | None = None,
        pred_argument: int | str | None = None,
        input_transforms: list[Callable] | None = None,
        target_transforms: list[Callable] | None = None,
        pred_transforms: list[Callable] | None = None,
    ):
        """
        Initializes FunctionAdapter with arguments and transforms for input, target, and prediction.

        Args:
            input_argument: The argument name for the input data.
            target_argument: The argument name for the target data.
            pred_argument: The argument name for the prediction data.
            input_transforms: A list of transforms to apply to the input data.
            target_transforms: A list of transforms to apply to the target data.
            pred_transforms: A list of transforms to apply to the prediction data.

        Raises:
            ValueError: If transforms are provided but the corresponding argument is None.
        """
        if input_argument is None and input_transforms is not None:
            raise ValueError("Input transforms provided but input_argument is None")
        if target_argument is None and target_transforms is not None:
            raise ValueError("Target transforms provided but target_argument is None")
        if pred_argument is None and pred_transforms is not None:
            raise ValueError("Pred transforms provided but pred_argument is None")

        self.input_argument = input_argument
        self.target_argument = target_argument
        self.pred_argument = pred_argument

        self.input_transforms = input_transforms
        self.target_transforms = target_transforms
        self.pred_transforms = pred_transforms

    def __call__(self, func: Callable, input: Any, target: Any, pred: Any) -> Any:
        """
        Applies the given function to the input, target, and prediction data.

        Args:
            func: The function to apply.
            input: The input data.
            target: The target data.
            pred: The prediction data.

        Returns:
            The result of the function call.
        """
        args = []
        kwargs = {}
        if self.input_argument is not None:
            input = self._transform(input, self.input_transforms)
            if isinstance(self.input_argument, int):
                args.insert(self.input_argument, input)
            else:
                kwargs[self.input_argument] = input

        if self.target_argument is not None:
            target = self._transform(target, self.target_transforms)
            if isinstance(self.target_argument, int):
                args.insert(self.target_argument, target)
            else:
                kwargs[self.target_argument] = target

        if self.pred_argument is not None:
            pred = self._transform(pred, self.pred_transforms)
            if isinstance(self.pred_argument, int):
                args.insert(self.pred_argument, pred)
            else:
                kwargs[self.pred_argument] = pred

        return func(*args, **kwargs)


class CriterionAdapter(FunctionAdapter):
    def __call__(self, criterion: Callable, input: Any, target: Any, pred: Any) -> Any:
        """
        Applies the criterion to the input, target, and prediction data.

        Args:
            criterion: The criterion (loss function) to apply.
            input: The input data.
            target: The target data.
            pred: The prediction data.

        Returns:
            The result of the criterion call.
        """
        return super().__call__(criterion, input, target, pred)


class MetricsAdapter(FunctionAdapter):
    """
    An adapter specifically for metrics calculations.
    """

    def __call__(self, metrics: Callable, input: Any, target: Any, pred: Any) -> Any:
        """
        Calculates metrics using the provided function and data.

        Args:
            metrics: The metrics function to apply.
            input: The input data.
            target: The target data.
            pred: The prediction data.

        Returns:
            The result of the metrics calculation.
        """
        return super().__call__(metrics, input, target, pred)


class LoggingAdapter(TransformAdapter):
    """
    An adapter for applying transformations to data before logging.
    """

    def __init__(
        self,
        input_transforms: list[Callable] | None = None,
        target_transforms: list[Callable] | None = None,
        pred_transforms: list[Callable] | None = None,
    ):
        """
        Initializes LoggingAdapter with transforms for input, target, and prediction.

        Args:
            input_transforms: A list of transforms to apply to the input data.
            target_transforms: A list of transforms to apply to the target data.
            pred_transforms: A list of transforms to apply to the prediction data.
        """

        self.input_transforms = input_transforms
        self.target_transforms = target_transforms
        self.pred_transforms = pred_transforms

    def input(self, data: Any):
        """
        Transforms the input data for logging.

        Args:
            data: The input data.

        Returns:
            The transformed input data.
        """
        return self._transform(data, self.input_transforms)

    def target(self, data: Any):
        """
        Transforms the target data for logging.

        Args:
            data: The target data.

        Returns:
            The transformed target data.
        """
        return self._transform(data, self.target_transforms)

    def pred(self, data: Any):
        """
        Transforms the prediction data for logging.

        Args:
            data: The prediction data.

        Returns:
            The transformed prediction data.
        """
        return self._transform(data, self.pred_transforms)
