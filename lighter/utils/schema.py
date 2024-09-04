from typing import Any, Callable, Dict, List, Optional, Union

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
    fit: Union[Dict[str, Any], str] = {}
    validate: Union[Dict[str, Any], str] = {}
    predict: Union[Dict[str, Any], str] = {}
    test: Union[Dict[str, Any], str] = {}
    lr_find: Union[Dict[str, Any], str] = {}
    scale_batch_size: Union[Dict[str, Any], str] = {}

    @model_validator(mode="after")
    def check_prohibited_args(self):
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
    requires: Optional[Any] = None
    project: Optional[str] = None
    vars: Dict[str, Any] = {}
    args: ArgsConfigSchema = ArgsConfigSchema()
    system: Dict[str, Any] = {}
    trainer: Dict[str, Any] = {}

    def __init__(self, **data):
        # Annoying workardound, Pydantic keeps all underscored keys private
        requires = data.pop("_requires_") if "_requires_" in data else None
        super().__init__(**data)
        self.requires = requires


# ----- LighterSystem schema -----


class DatasetSchema(BaseModel):
    train: Optional[Dataset] = None
    val: Optional[Dataset] = None
    test: Optional[Dataset] = None
    predict: Optional[Dataset] = None


class SamplerSchema(BaseModel):
    train: Optional[Sampler] = None
    val: Optional[Sampler] = None
    test: Optional[Sampler] = None
    predict: Optional[Sampler] = None


class CollateFnSchema(BaseModel):
    train: Optional[Callable] = None
    val: Optional[Callable] = None
    test: Optional[Callable] = None
    predict: Optional[Callable] = None


class MetricsSchema(BaseModel):
    train: Optional[Union[Any, List[Any], Dict[str, Any]]] = None
    val: Optional[Union[Any, List[Any], Dict[str, Any]]] = None
    test: Optional[Union[Any, List[Any], Dict[str, Any]]] = None

    @model_validator(mode="after")
    def setup_metrics(self):
        for field in self.model_fields:
            if getattr(self, field) is not None:
                setattr(self, field, MetricCollection(getattr(self, field)))
        return self


class ModeSchema(BaseModel):
    train: Optional[Union[Callable, List[Callable]]] = None
    val: Optional[Union[Callable, List[Callable]]] = None
    test: Optional[Union[Callable, List[Callable]]] = None
    predict: Optional[Union[Callable, List[Callable]]] = None


class DataSchema(BaseModel):
    input: Optional[Union[Callable, List[Callable]]] = None
    target: Optional[Union[Callable, List[Callable]]] = None
    pred: Optional[Union[Callable, List[Callable]]] = None


class PostprocessingSchema(BaseModel):
    batch: ModeSchema = ModeSchema()
    criterion: DataSchema = DataSchema()
    metrics: DataSchema = DataSchema()
    logging: DataSchema = DataSchema()
