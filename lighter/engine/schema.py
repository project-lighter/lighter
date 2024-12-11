"""
This module defines schemas for configuration validation using Pydantic, ensuring correct structure and types.
"""

from typing import Any, Callable, List

import warnings

from pydantic import BaseModel, model_validator
from torch.utils.data import Dataset, Sampler
from torchmetrics import MetricCollection

warnings.filterwarnings("ignore", 'Field name "validate" in "ArgsConfigSchema" shadows an attribute in parent "BaseModel"')

BaseModel.model_config["arbitrary_types_allowed"] = True
BaseModel.model_config["hide_input_in_errors"] = True
BaseModel.model_config["extra"] = "forbid"
# ----- Config schema -----


class ArgsConfigSchema(BaseModel):
    """
    Schema for validating arguments configuration, ensuring correct types and prohibiting certain keys.
    """

    fit: dict[str, Any] | str = {}
    validate: dict[str, Any] | str = {}
    predict: dict[str, Any] | str = {}
    test: dict[str, Any] | str = {}
    lr_find: dict[str, Any] | str = {}
    scale_batch_size: dict[str, Any] | str = {}

    @model_validator(mode="after")
    def check_prohibited_args(self):
        """
        Validates that prohibited arguments are not present in the configuration.

        Raises:
            ValueError: If prohibited arguments are found.
        """
        prohibited_keys = ["model", "train_loaders", "validation_loaders", "dataloaders", "datamodule"]
        for field in self.model_fields:
            field_value = getattr(self, field)
            if isinstance(field_value, dict):
                found_keys = [key for key in prohibited_keys if key in field_value]
                if found_keys:
                    raise ValueError(
                        f"Found the following prohibited argument(s) in 'args#{field}': "
                        f"{found_keys}. Model and datasets should be defined within the 'system'."
                    )
            elif isinstance(field_value, str) and not (field_value.startswith("%") or field_value.startswith("@")):
                raise ValueError(f"Only dict or interpolators starting with '%' or '@' are allowed for 'args#{field}'.")
        return self


class ConfigSchema(BaseModel):
    """
    Schema for validating the overall configuration, including system and trainer settings.
    """

    requires: Any | None = None
    project: str | None = None
    vars: dict[str, Any] = {}
    args: ArgsConfigSchema = ArgsConfigSchema()
    system: dict[str, Any] = {}
    trainer: dict[str, Any] = {}

    def __init__(self, **data):
        # Annoying workardound, Pydantic keeps all underscored keys private
        requires = data.pop("_requires_") if "_requires_" in data else None
        super().__init__(**data)
        self.requires = requires


# ----- LighterSystem schema -----


class DatasetSchema(BaseModel):
    """
    Schema for validating dataset configurations for different stages (train, val, test, predict).
    """

    train: Dataset | None = None
    val: Dataset | None = None
    test: Dataset | None = None
    predict: Dataset | None = None


class SamplerSchema(BaseModel):
    """
    Schema for validating sampler configurations for different stages (train, val, test, predict).
    """

    train: Sampler | None = None
    val: Sampler | None = None
    test: Sampler | None = None
    predict: Sampler | None = None


class CollateFnSchema(BaseModel):
    """
    Schema for validating collate function configurations for different stages (train, val, test, predict).
    """

    train: Callable | None = None
    val: Callable | None = None
    test: Callable | None = None
    predict: Callable | None = None


class MetricsSchema(BaseModel):
    """
    Schema for validating metrics configurations, supporting single or multiple metrics.
    """

    train: Any | List[Any] | dict[str, Any] | None = None
    val: Any | List[Any] | dict[str, Any] | None = None
    test: Any | List[Any] | dict[str, Any] | None = None

    @model_validator(mode="after")
    def setup_metrics(self):
        """
        Converts metrics into a MetricCollection for consistent handling.

        Returns:
            self: The updated schema with MetricCollections.
        """
        for field in self.model_fields:
            if getattr(self, field) is not None:
                setattr(self, field, MetricCollection(getattr(self, field)))
        return self


class ModeSchema(BaseModel):
    """
    Schema for validating mode-specific configurations, such as postprocessing functions.
    """

    train: Callable | List[Callable] | None = None
    val: Callable | List[Callable] | None = None
    test: Callable | List[Callable] | None = None
    predict: Callable | List[Callable] | None = None


class DataSchema(BaseModel):
    """
    Schema for validating data-specific configurations, such as input, target, and prediction processing.
    """

    input: Callable | List[Callable] | None = None
    target: Callable | List[Callable] | None = None
    pred: Callable | List[Callable] | None = None


class PostprocessingSchema(BaseModel):
    """
    Schema for validating postprocessing configurations, ensuring correct structure for batch, criterion, metrics, and logging.
    """

    batch: ModeSchema = ModeSchema()
    criterion: DataSchema = DataSchema()
    metrics: DataSchema = DataSchema()
    logging: DataSchema = DataSchema()
