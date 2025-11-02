from typing import Any, Callable, Optional, Union

from torch.nn import Module
from torchmetrics import MetricCollection

from lighter.utils.types.enums import Data


class Flow:
    """
    The Flow defines the entire step logic, from unpacking the batch to defining the output.
    It follows a "convention over configuration" philosophy. The output of the model is always
    stored in the context as 'pred', and the output of the criterion is always stored as 'loss'.
    """

    def __init__(
        self,
        batch: Union[dict[str, Any], list[str]],
        model: Optional[Union[dict[str, Any], list[Any]]] = None,
        criterion: Optional[Union[dict[str, Any], list[Any]]] = None,
        metrics: Optional[Union[dict[str, Any], list[Any]]] = None,
        output: Optional[dict[str, Any]] = None,
        logging: Optional[dict[str, Any]] = None,
    ):
        self.batch_config = batch
        self.model_config = model or {}
        self.criterion_config = criterion or {}
        self.metrics_config = metrics or {}
        self.output_config = output or {}
        self.logging_config = logging or {}

    def __call__(
        self,
        batch: Any,
        model: Module,
        criterion: Optional[Callable] = None,
        metrics: Optional[MetricCollection] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # 1. Unpack the batch data into a new context dictionary
        context = self._unpack_batch(batch)

        # 2. Run the model and add its prediction to the context
        self._run_model(context, model)

        # 3. Run the criterion (if provided) and add the loss to the context
        self._run_criterion(context, criterion)

        # 4. Update the metrics (if provided)
        self._run_metrics(context, metrics)

        # 5. Apply logging transformations to produce the final context for output
        final_context = self._apply_logging_transforms(context)

        # 6. Build the final output dictionary from the context
        output = self._build_output(final_context)

        return output

    # --- Private Helper Methods ---

    def _unpack_batch(self, batch: Any) -> dict[str, Any]:
        """Handles the logic for the 'batch' configuration."""
        context: dict[str, Any] = {}
        if isinstance(self.batch_config, dict):
            for key, accessor in self.batch_config.items():
                if callable(accessor):
                    context[key] = accessor(batch)
                elif isinstance(accessor, str):
                    try:
                        context[key] = batch[accessor]
                    except (KeyError, TypeError) as e:
                        raise ValueError(f"Could not access '{accessor}' from batch.\n{e}") from e
                else:
                    raise TypeError(f"Unsupported accessor type: {type(accessor)}")
        elif isinstance(self.batch_config, list):
            for i, key in enumerate(self.batch_config):
                try:
                    context[key] = batch[i]
                except IndexError as e:
                    raise ValueError(f"Could not access index {i} from batch.\n{e}") from e
        else:
            raise TypeError(f"Unsupported batch config type: {type(self.batch_config)}")
        return context

    def _run_model(self, context: dict[str, Any], model: Module) -> None:
        """Handles the logic for the 'model' configuration."""
        model_args, model_kwargs = self._prepare_args_kwargs(context, self.model_config)
        context[Data.PRED] = model(*model_args, **model_kwargs)

    def _run_criterion(self, context: dict[str, Any], criterion: Optional[Callable]) -> None:
        """Handles the logic for the 'criterion' configuration."""
        if criterion and self.criterion_config:
            criterion_args, criterion_kwargs = self._prepare_args_kwargs(context, self.criterion_config)
            context[Data.LOSS] = criterion(*criterion_args, **criterion_kwargs)

    def _run_metrics(self, context: dict[str, Any], metrics: Optional[MetricCollection]) -> None:
        """Handles the logic for the 'metrics' configuration."""
        if metrics and self.metrics_config:
            metrics_args, metrics_kwargs = self._prepare_args_kwargs(context, self.metrics_config)
            metrics.update(*metrics_args, **metrics_kwargs)
            context[Data.METRICS] = metrics

    def _apply_logging_transforms(self, context: dict[str, Any]) -> dict[str, Any]:
        """Handles the logic for the 'logging' configuration."""
        logging_context = context.copy()
        if self.logging_config:
            for key, transform_key in self.logging_config.items():
                logging_context[key] = self._get_value(context, transform_key)
        return logging_context

    def _build_output(self, context: dict[str, Any]) -> dict[str, Any]:
        """Handles the logic for the 'output' configuration."""
        return {arg: self._get_value(context, key) for arg, key in self.output_config.items()}

    def _get_value(self, context: dict[str, Any], key: Union[str, Callable, list]) -> Any:
        """Resolves a value from the context given a key, which can be a string, a callable, or a list of callables for a pipeline."""
        if isinstance(key, list):
            value = self._get_value(context, key[0])
            for transform in key[1:]:
                value = transform(value)
            return value

        if callable(key):
            return key(context)

        if isinstance(key, str):
            if "." in key:
                value = context
                for k in key.split("."):
                    try:
                        if isinstance(value, dict):
                            value = value[k]
                        else:
                            value = getattr(value, k)
                    except (KeyError, AttributeError) as e:
                        raise KeyError(f"Could not resolve nested key '{key}' from context.\n{e}") from e
                return value
            if key in context:
                return context[key]
            else:
                return key

        raise TypeError(f"Unsupported key type: {type(key)}")

    def _prepare_args_kwargs(
        self, context: dict[str, Any], config: Union[dict[str, Any], list[Any]]
    ) -> tuple[list[Any], dict[str, Any]]:
        """Prepares args and kwargs by resolving values from the context based on the given config."""
        if isinstance(config, dict):
            kwargs = {arg: self._get_value(context, key) for arg, key in config.items()}
            return [], kwargs
        if isinstance(config, list):
            args = [self._get_value(context, key) for key in config]
            return args, {}
        return [], {}

    @staticmethod
    def get_default(mode: str) -> "Flow":
        if mode in ["train", "val"]:
            return Flow(
                batch=[Data.INPUT, Data.TARGET],
                model=[Data.INPUT],
                criterion=[Data.PRED, Data.TARGET],
                metrics=[Data.PRED, Data.TARGET],
                output={Data.PRED: Data.PRED, Data.LOSS: Data.LOSS},
            )
        elif mode == "test":
            return Flow(
                batch=[Data.INPUT, Data.TARGET],
                model=[Data.INPUT],
                metrics=[Data.PRED, Data.TARGET],
                output={Data.PRED: Data.PRED},
            )
        elif mode == "predict":
            return Flow(
                batch={Data.INPUT: lambda batch: batch},
                model=[Data.INPUT],
                output={Data.PRED: Data.PRED},
            )
        else:
            raise ValueError(f"Invalid mode for default Flow: {mode}")
