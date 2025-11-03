from typing import Any, Callable

from torch.nn import Module
from torchmetrics import MetricCollection

from lighter.utils.types.enums import Data


class Flow:
    """
    The Flow defines the entire step logic, from unpacking the batch to defining the output.
    It follows a "convention over configuration" philosophy. The output of the model is always
    stored in the data as 'pred', and the output of the criterion is always stored as 'loss'.
    """

    def __init__(
        self,
        batch: dict[str, Any] | list[str],
        model: dict[str, Any] | list[Any] | str | None = None,
        criterion: dict[str, Any] | list[Any] | None = None,
        metrics: dict[str, Any] | list[Any] | None = None,
        output: dict[str, Any] | None = None,
    ):
        self.batch_config = batch
        self.model_config = model or {}
        self.criterion_config = criterion or {}
        self.metrics_config = metrics or {}
        self.output_config = output or {}

    def __call__(
        self,
        batch: Any,
        model: Module,
        criterion: Callable | None = None,
        metrics: MetricCollection | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # 1. Unpack the batch data into a new data dictionary
        data = self._unpack_batch(batch, data)

        # 2. Run the model and add its prediction to the data
        data = self._run_model(data, model)

        # 3. Run the criterion (if provided) and add the loss to the data
        data = self._run_criterion(data, criterion)

        # 4. Update the metrics (if provided)
        data = self._run_metrics(data, metrics)

        # 5. Run the output transforms and add them to the data
        data = self._run_output(data)

        return data

    # --- Private Helper Methods ---

    def _unpack_batch(self, batch: Any, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Handles the logic for the 'batch' configuration."""
        data = data or {}
        if isinstance(self.batch_config, dict):
            for key, accessor in self.batch_config.items():
                if callable(accessor):
                    data[key] = accessor(batch)
                elif isinstance(accessor, str):
                    try:
                        data[key] = batch[accessor]
                    except (KeyError, TypeError) as e:
                        raise ValueError(f"Could not access '{accessor}' from batch.\n{e}") from e
                else:
                    raise TypeError(f"Unsupported accessor type: {type(accessor)}")
        elif isinstance(self.batch_config, list):
            for i, key in enumerate(self.batch_config):
                try:
                    data[key] = batch[i]
                except IndexError as e:
                    raise ValueError(f"Could not access index {i} from batch.\n{e}") from e
        else:
            raise TypeError(f"Unsupported batch config type: {type(self.batch_config)}")
        return data

    def _run_model(self, data: dict[str, Any], model: Module) -> dict[str, Any]:
        """Handles the logic for the 'model' configuration."""
        model_args, model_kwargs = self._prepare_args_kwargs(data, self.model_config)
        data[Data.PRED] = model(*model_args, **model_kwargs)
        return data

    def _run_criterion(self, data: dict[str, Any], criterion: Callable | None) -> dict[str, Any]:
        """Handles the logic for the 'criterion' configuration."""
        if criterion and self.criterion_config:
            criterion_args, criterion_kwargs = self._prepare_args_kwargs(data, self.criterion_config)
            data[Data.LOSS] = criterion(*criterion_args, **criterion_kwargs)
        return data

    def _run_metrics(self, data: dict[str, Any], metrics: MetricCollection | None) -> dict[str, Any]:
        """Handles the logic for the 'metrics' configuration."""
        if metrics and self.metrics_config:
            metrics_args, metrics_kwargs = self._prepare_args_kwargs(data, self.metrics_config)
            metrics.update(*metrics_args, **metrics_kwargs)
            data[Data.METRICS] = metrics
        return data

    def _run_output(self, data: dict[str, Any]) -> dict[str, Any]:
        """Handles the logic for the 'output' configuration."""
        if self.output_config:
            for key, transform_key in self.output_config.items():
                data[key] = self._get_value(data, transform_key)
        return data

    def _get_value(self, data: dict[str, Any], key: str | Callable | list) -> Any:
        """
        Resolves a value from the data given a key, which can be a string, a callable,
        or a list of callables for a pipeline.
        """
        if isinstance(key, list):
            value = self._get_value(data, key[0])
            for transform in key[1:]:
                value = transform(value)
            return value

        if callable(key):
            return key(data)

        if isinstance(key, str):
            if "." in key:
                value = data
                for k in key.split("."):
                    try:
                        if isinstance(value, dict):
                            value = value[k]
                        else:
                            value = getattr(value, k)
                    except (KeyError, AttributeError) as e:
                        raise KeyError(f"Could not resolve nested key '{key}' from data.\n{e}") from e
                return value
            # A key not found in the data should raise an error.
            if key not in data:
                raise KeyError(f"Key '{key}' not found in the data.")
            return data[key]

        raise TypeError(f"Unsupported key type: {type(key)}")

    def _prepare_args_kwargs(
        self, data: dict[str, Any], config: dict[str, Any] | list[Any] | str
    ) -> tuple[list[Any], dict[str, Any]]:
        """Prepares args and kwargs by resolving values from the data based on the given config."""
        if isinstance(config, dict):
            kwargs = {arg: self._get_value(data, key) for arg, key in config.items()}
            return [], kwargs
        if isinstance(config, list):
            args = [self._get_value(data, key) for key in config]
            return args, {}
        if isinstance(config, str):
            args = [self._get_value(data, config)]
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
            )
        elif mode == "test":
            return Flow(
                batch=[Data.INPUT, Data.TARGET],
                model=[Data.INPUT],
                metrics=[Data.PRED, Data.TARGET],
            )
        elif mode == "predict":
            return Flow(
                batch={Data.INPUT: lambda batch: batch[0] if isinstance(batch, (list, tuple)) else batch},
                model=[Data.INPUT],
            )
        else:
            raise ValueError(f"Invalid mode for default Flow: {mode}")
