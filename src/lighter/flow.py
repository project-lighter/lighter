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
        model: dict[str, Any] | list[Any] | None = None,
        criterion: dict[str, Any] | list[Any] | None = None,
        metrics: dict[str, Any] | list[Any] | None = None,
        output: dict[str, Any] | None = None,
    ):
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
        # 1. Initialize data dictionary and add the batch to it.
        data = data or {}
        data["batch"] = batch

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
            # Bracket notation for batch access, e.g., 'batch[0]' or 'batch["input"]'
            if key.startswith("batch["):
                try:
                    value = data["batch"]
                    import re

                    accessors = re.findall(r"\[(.*?)\]", key)
                    for acc in accessors:
                        acc = acc.strip()
                        if (acc.startswith('"') and acc.endswith('"')) or (acc.startswith("'") and acc.endswith("'")):
                            value = value[acc[1:-1]]
                        elif acc.isdigit() or (acc.startswith("-") and acc[1:].isdigit()):
                            value = value[int(acc)]
                        else:
                            raise ValueError(f"Unsupported accessor '{acc}' in key '{key}'")
                    return value
                except (KeyError, IndexError, ValueError) as e:
                    raise KeyError(f"Could not resolve nested key '{key}' from data.\n{e}") from e

            if "." in key:
                value = data
                for k in key.split("."):
                    try:
                        if isinstance(value, dict):
                            value = value[k]
                        elif isinstance(value, (list, tuple)) and k.isdigit():
                            value = value[int(k)]
                        else:
                            value = getattr(value, k)
                    except (KeyError, AttributeError, IndexError) as e:
                        raise KeyError(f"Could not resolve nested key '{key}' from data.\n{e}") from e
                return value
            # A key not found in the data should raise an error.
            if key not in data:
                raise KeyError(f"Key '{key}' not found in the data.")
            return data[key]

        raise TypeError(f"Unsupported key type: {type(key)}")

    def _prepare_args_kwargs(
        self, data: dict[str, Any], config: dict[str, Any] | list[Any]) -> tuple[list[Any], dict[str, Any]]:

        """Prepares args and kwargs by resolving values from the data based on the given config."""

        if isinstance(config, dict):

            kwargs = {arg: self._get_value(data, key) for arg, key in config.items()}
            return [], kwargs

        if isinstance(config, list):
            args = [self._get_value(data, key) for key in config]
            return args, {}

        raise TypeError(f"Flow configuration must either be a list (for positional args) or a dict (for keyword args), but got {type(config)}.")

    @staticmethod
    def get_default(mode: str) -> "Flow":
        if mode in ["train", "val"]:
            return Flow(
                model=["batch.0"],
                criterion=["pred", "batch.1"],
                metrics=["pred", "batch.1"],
            )
        elif mode == "test":
            return Flow(
                model=["batch.0"],
                metrics=["pred", "batch.1"],
            )
        elif mode == "predict":
            return Flow(
                model=[lambda data: data["batch"][0] if isinstance(data["batch"], (list, tuple)) else data["batch"]],
            )
        else:
            raise ValueError(f"Invalid mode for default Flow: {mode}")