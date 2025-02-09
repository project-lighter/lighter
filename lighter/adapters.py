from typing import Any, Callable

from lighter.utils.misc import ensure_list


class _TransformsAdapter:
    """
    Adapter for applying transformations to data.

    Args:
        input_transforms: A single or a list of transforms to apply to the input data.
        target_transforms: A single or a list of transforms to apply to the target data.
        pred_transforms: A single or a list of transforms to apply to the prediction data.
    """

    def __init__(
        self,
        input_transforms: Callable | list[Callable] | None = None,
        target_transforms: Callable | list[Callable] | None = None,
        pred_transforms: Callable | list[Callable] | None = None,
    ):
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms
        self.pred_transforms = pred_transforms

    def __call__(self, input: Any, target: Any, pred: Any) -> tuple[Any, Any, Any]:
        """
        Applies the specified transforms to the input, target, and prediction data.

        Args:
            input: The input data.
            target: The target data.
            pred: The prediction data.

        Returns:
            The transformed (input, target, prediction) data.
        """
        input = self._transform(input, self.input_transforms)
        target = self._transform(target, self.target_transforms)
        pred = self._transform(pred, self.pred_transforms)
        return input, target, pred

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


class _ArgumentsAdapter:
    """
    Base adapter for adapting arguments to a function based on specified argument names or positions.
    """

    def __init__(
        self,
        input_argument: int | str | None = None,
        target_argument: int | str | None = None,
        pred_argument: int | str | None = None,
    ):
        # Ensure that the positionals are consecutive integers.
        # There cannot be positional 0 and 2, without 1. Same with a positional 1 without 0.
        positionals = sorted(arg for arg in (input_argument, target_argument, pred_argument) if isinstance(arg, int))
        if positionals != list(range(len(positionals))):
            raise ValueError("Positional arguments must be consecutive integers starting from 0.")

        self.input_argument = input_argument
        self.target_argument = target_argument
        self.pred_argument = pred_argument

    def __call__(self, input: Any, target: Any, pred: Any) -> tuple[list[Any], dict[str, Any]]:
        """
        Adapts the input, target, and prediction data to the specified argument positions or names.

        Args:
            input: The input data to be adapted.
            target: The target data to be adapted.
            pred: The prediction data to be adapted.

        Returns:
            A tuple containing a list of positional arguments and a dictionary of keyword arguments.
        """
        args = []  # List to store positional arguments
        kwargs = {}  # Dictionary to store keyword arguments

        # Mapping of argument names to their respective values
        argument_map = {"input_argument": input, "target_argument": target, "pred_argument": pred}

        # Iterate over the argument map to adapt arguments
        for arg_name, value in argument_map.items():
            # Get the position or name of the argument from the instance attributes
            arg_position = getattr(self, arg_name)
            if arg_position is not None:
                if isinstance(arg_position, int):
                    # Insert the value into the args list at the specified position
                    args.insert(arg_position, value)
                elif isinstance(arg_position, str):
                    # Add the value to the kwargs dictionary with the specified name
                    kwargs[arg_position] = value
                else:
                    # Raise an error if the argument type is invalid
                    raise ValueError(f"Invalid {arg_name} type: {type(arg_position)}")

        # Return the adapted positional and keyword arguments
        return args, kwargs


class _ArgumentsAndTransformsAdapter(_ArgumentsAdapter, _TransformsAdapter):
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
        Initializes the Arguments and Transforms Adapter.

        Args:
            input_argument: Position or name for the input data.
            target_argument: Position or name for the target data.
            pred_argument: Position or name for the prediction data.
            input_transforms: Transforms to apply to the input data.
            target_transforms: Transforms to apply to the target data.
            pred_transforms: Transforms to apply to the prediction data.

        Raises:
            ValueError: If transforms are provided without corresponding argument specifications.
        """
        # Validate transform arguments
        if input_argument is None and input_transforms is not None:
            raise ValueError("Input transforms provided but input_argument is None")
        if target_argument is None and target_transforms is not None:
            raise ValueError("Target transforms provided but target_argument is None")
        if pred_argument is None and pred_transforms is not None:
            raise ValueError("Pred transforms provided but pred_argument is None")

        _ArgumentsAdapter.__init__(self, input_argument, target_argument, pred_argument)
        _TransformsAdapter.__init__(self, input_transforms, target_transforms, pred_transforms)

    def __call__(self, fn: Callable, input: Any, target: Any, pred: Any) -> Any:
        """
        Applies transforms and adapts arguments before calling the provided function.

        Args:
            fn: The function/method to be called (e.g., a loss function or metric).
            input: The input data.
            target: The target data.
            pred: The prediction data.

        Returns:
            The result of the function call.
        """
        # Apply the transforms to the input, target, and prediction data
        input, target, pred = _TransformsAdapter.__call__(self, input, target, pred)
        # Map the input, target, and prediction data to the function arguments
        args, kwargs = _ArgumentsAdapter.__call__(self, input, target, pred)
        # Call the provided function with the adapted arguments
        return fn(*args, **kwargs)


class BatchAdapter:
    def __init__(
        self,
        input_accessor: int | str | Callable,
        target_accessor: int | str | Callable | None = None,
        identifier_accessor: int | str | Callable | None = None,
    ):
        """
        Initializes BatchAdapter with accessors for input, target, and identifier.

        Args:
            input_accessor: Accessor for the identifier data. Can be an index (lists/tuples), a key (dictionaries),
                a callable (custom batch processing).
            target_accessor: Accessor for the target data. Can be an index (for lists/tuples),
                             a key (for dictionaries), or a callable (for custom batch processing).
            identifier_accessor: Accessor for the identifier data. Can be an index (lists/tuples), a key (dictionaries),
                a callable (custom batch processing), or None if no identifier is present.
        """
        self.input_accessor = input_accessor
        self.target_accessor = target_accessor
        self.identifier_accessor = identifier_accessor

    def __call__(self, batch: Any) -> tuple[Any, Any, Any]:
        """
        Accesses the identifier, input, and target data from the batch.

        Args:
            batch: The batch data from which to extract information.

        Returns:
            A tuple containing (identifier, input, target).

        Raises:
            ValueError: If accessors are invalid for the provided batch structure.
        """
        input = self._access_value(batch, self.input_accessor)
        target = self._access_value(batch, self.target_accessor)
        identifier = self._access_value(batch, self.identifier_accessor)
        return input, target, identifier

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
            return None
        elif isinstance(accessor, int) and isinstance(data, (tuple, list)):
            return data[accessor]
        elif isinstance(accessor, str) and isinstance(data, dict):
            return data[accessor]
        elif callable(accessor):
            return accessor(data)
        else:
            raise ValueError(f"Invalid accessor {accessor} of type {type(accessor)} for data type {type(data)}.")


class CriterionAdapter(_ArgumentsAndTransformsAdapter):
    """
    This adapter processes and transforms the input, target, and prediction data, if specified,
    and forwards them to the specified arguments of a criterion (loss function).
    """

    def __call__(self, criterion: Callable, input: Any, target: Any, pred: Any) -> Any:
        """
        Applies transforms and adapts arguments before calling the provided metric function.

        Args:
            criterion: The criterion (loss function).
            input: The input data to transform with `input_transforms` if specified and pass to the metric with
                the position or argument name specified by `input_argument`.
            target: The target data to transform with `target_transforms` if specified and pass to the metric with
                the position or argument name specified by `target_argument`.
            pred: The prediction data to transform with `pred_transforms` if specified and pass to the metric with
                the position or argument name specified by `pred_argument`.

        Returns:
            The result of the metric function call.
        """
        return super().__call__(criterion, input, target, pred)


class MetricsAdapter(_ArgumentsAndTransformsAdapter):
    """
    This adapter processes and transforms the input, target, and prediction data, if specified,
    and forwards them to the specified arguments of a metric.
    """

    def __call__(self, metric: Callable, input: Any, target: Any, pred: Any) -> Any:
        """
        Applies transforms and adapts arguments before calling the provided metric function.

        Args:
            metric: The metric.
            input: The input data to transform with `input_transforms` if specified and pass to the metric with
                the position or argument name specified by `input_argument`.
            target: The target data to transform with `target_transforms` if specified and pass to the metric with
                the position or argument name specified by `target_argument`.
            pred: The prediction data to transform with `pred_transforms` if specified and pass to the metric with
                the position or argument name specified by `pred_argument`.

        Returns:
            The result of the metric function call.
        """
        return super().__call__(metric, input, target, pred)


class LoggingAdapter(_TransformsAdapter):
    """
    Adapter for applying logging transformations to data.

    This adapter handles the transformation of input, target, and prediction data
    specifically for logging purposes. It can preprocess or format the data before
    logging, ensuring consistency and readability in logs.

    """

    def __init__(
        self,
        input_transforms: list[Callable] | None = None,
        target_transforms: list[Callable] | None = None,
        pred_transforms: list[Callable] | None = None,
    ):
        super().__init__(input_transforms, target_transforms, pred_transforms)
