from collections.abc import Callable
from typing import Any

from torch.nn import Module
from torchmetrics import MetricCollection

from lighter.utils.resolver import resolve_value
from lighter.utils.types.enums import Data


class Flow:
    """
    Defines the step logic for lighter.System, from batch unpacking to output.

    Follows a "convention over configuration" philosophy. Model output is stored
    as `Data.PRED` ('pred'), and criterion output as `Data.LOSS` ('loss').
    """

    def __init__(
        self,
        model: dict[str, Any] | list[Any] | None = None,
        criterion: dict[str, Any] | list[Any] | None = None,
        metrics: dict[str, Any] | list[Any] | None = None,
        output: dict[str, Any] | None = None,
    ):
        """
        Initializes a Flow with configurations for model, criterion, metrics, and output.

        Args:
            model: Config for model inputs. Can be a list (positional args) or dict (keyword args).
                   Values are keys to be resolved from the `data` dictionary.
                   Example: `model=["batch.0"]` or `model={"input": "batch.0"}`
            criterion: Config for loss function inputs. Similar to `model`.
                       Example: `criterion=["pred", "batch.1"]`
            metrics: Config for metric updates. Similar to `model`.
                     Example: `metrics={"preds": "pred", "target": "batch.1"}`
            output: Config for final output transformations. A dictionary where keys are
                    new keys in `data`, and values are keys to be resolved.
                    Example: `output={"final_loss": "loss.total"}`
        """
        self.model_config = model or {}
        self.criterion_config = criterion or {}
        self.metrics_config = metrics or {}
        self.output_config = output or {}

    def __call__(
        self,
        batch: Any,
        model: Module,
        criterion: Callable[..., Any] | None = None,
        metrics: MetricCollection | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Executes a single step (e.g., train, validation, test, predict).

        Orchestrates data flow:
        1. Initializes `data` with the `batch`.
        2. Runs `model`, stores predictions (`Data.PRED`).
        3. Runs `criterion` (if provided), stores loss (`Data.LOSS`).
        4. Updates `metrics` (if provided).
        5. Applies `output` transformations.

        Args:
            batch: Input data batch.
            model: Neural network model.
            criterion: Loss function (optional).
            metrics: Metric collection (optional).
            data: Optional initial data dictionary.

        Returns:
            Updated `data` dictionary with predictions, loss, and metrics.
        """
        # 1. Initialize data dictionary and add the batch to it.
        data = data or {}
        data["batch"] = batch

        # 2. Run the model and add its prediction to the data.
        data = self._run_model(data, model)

        # 3. Run the criterion (if provided) and add the loss to the data.
        data = self._run_criterion(data, criterion)

        # 4. Update the metrics (if provided).
        data = self._run_metrics(data, metrics)

        # 5. Run the output transforms and add them to the data.
        data = self._run_output(data)

        return data

    # --- Private Helper Methods ---

    def _run_model(self, data: dict[str, Any], model: Module) -> dict[str, Any]:
        """
        Runs the model using `self.model_config` and stores predictions.

        Args:
            data: Current data dictionary.
            model: Neural network model.

        Returns:
            Updated `data` with model predictions (`Data.PRED`).
        """
        model_args, model_kwargs = self._prepare_args_kwargs(data, self.model_config)
        data[Data.PRED] = model(*model_args, **model_kwargs)
        return data

    def _run_criterion(self, data: dict[str, Any], criterion: Callable[..., Any] | None) -> dict[str, Any]:
        """
        Runs the criterion (loss function) using `self.criterion_config` and stores loss.

        Args:
            data: Current data dictionary.
            criterion: Loss function.

        Returns:
            Updated `data` with computed loss (`Data.LOSS`).
        """
        if criterion and self.criterion_config:
            criterion_args, criterion_kwargs = self._prepare_args_kwargs(data, self.criterion_config)
            data[Data.LOSS] = criterion(*criterion_args, **criterion_kwargs)
        return data

    def _run_metrics(self, data: dict[str, Any], metrics: MetricCollection | None) -> dict[str, Any]:
        """
        Updates metrics using `self.metrics_config`.

        Args:
            data: Current data dictionary.
            metrics: Metric collection.

        Returns:
            Updated `data` with the metrics collection (`Data.METRICS`).
        """
        if metrics and self.metrics_config:
            metrics_args, metrics_kwargs = self._prepare_args_kwargs(data, self.metrics_config)
            metrics.update(*metrics_args, **metrics_kwargs)
            data[Data.METRICS] = metrics
        return data

    def _run_output(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Applies output transformations defined in `self.output_config`.

        Resolves values from a copy of `data` (to avoid overwrites) and stores
        transformed values under new keys in `data`.

        Args:
            data: Current data dictionary.

        Returns:
            Updated `data` with transformed output values.
        """
        if self.output_config:
            # Create a copy of the data dictionary to ensure that the resolver always operates on the original data.
            # This prevents issues where a resolved value overwrites a key that is then needed for subsequent resolutions.
            original_data = data.copy()
            for key, transform_key in self.output_config.items():
                data[key] = resolve_value(original_data, transform_key)
        return data

    def _prepare_args_kwargs(
        self, data: dict[str, Any], config: dict[str, Any] | list[Any]
    ) -> tuple[list[Any], dict[str, Any]]:
        """
        Prepares `args` and `kwargs` by resolving values from `data` based on `config`.

        - If `config` is a dict, it's treated as `kwargs`.
        - If `config` is a list, it's treated as `args`.

        Args:
            data: Current data dictionary.
            config: Configuration for arguments (dict for kwargs, list for args).
                    Values in config are keys to be resolved from `data`.

        Returns:
            A tuple: (`list` of positional arguments, `dict` of keyword arguments).

        Raises:
            TypeError: If `config` is neither a list nor a dictionary.
        """
        if isinstance(config, dict):
            kwargs = {arg: resolve_value(data, key) for arg, key in config.items()}
            return [], kwargs

        if isinstance(config, list):
            args = [resolve_value(data, key) for key in config]
            return args, {}

        raise TypeError(
            f"Flow configuration must either be a list (for positional args) or a dict (for keyword args), but got {type(config)} {config}."
        )

    @staticmethod
    def get_default(mode: str) -> "Flow":
        """
        Provides default `Flow` configurations for common operational modes.

        Args:
            mode: The operational mode ('train', 'val', 'test', 'predict').

        Returns:
            A `Flow` instance with default settings for the specified mode.

        Raises:
            ValueError: If an invalid mode is provided.
        """
        if mode in ["train", "val"]:
            # Default flow for training and validation, including model, criterion, and metrics.
            return Flow(
                model=["batch.0"],
                criterion=["pred", "batch.1"],
                metrics=["pred", "batch.1"],
            )
        elif mode == "test":
            # Default flow for testing, including model and metrics.
            return Flow(
                model=["batch.0"],
                metrics=["pred", "batch.1"],
            )
        elif mode == "predict":
            # Default flow for prediction, only running the model.
            # Uses a lambda to handle batch unpacking for prediction.
            return Flow(
                model=[lambda data: data["batch"][0] if isinstance(data["batch"], (list, tuple)) else data["batch"]],
            )
        else:
            raise ValueError(f"Invalid mode for default Flow: {mode}")
