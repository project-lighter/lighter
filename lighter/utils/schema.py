from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, root_validator
from torch.utils.data import DataLoader, Dataset, Sampler
from torchmetrics import MetricCollection

BaseModel.model_config["arbitrary_types_allowed"] = True
BaseModel.model_config["hide_input_in_errors"] = True


class SubscriptableModel:
    def __getitem__(self, item):
        return getattr(self, item)


# ----- Config schema -----


class ArgsConfigSchema(BaseModel):
    fit: Dict[str, Any] = {}
    validate: Dict[str, Any] = {}
    predict: Dict[str, Any] = {}
    lr_find: Dict[str, Any] = {}
    scale_batch_size: Dict[str, Any] = {}

    @root_validator(skip_on_failure=True)
    def check_prohibited_args(cls, fields):  # pylint: disable=no-self-argument
        prohibited_keys = ["model", "train_loaders", "validation_loaders", "dataloaders", "datamodule"]
        for method, args in fields.items():
            found_keys = [key for key in prohibited_keys if key in args]
            if found_keys:
                raise ValueError(
                    f"Found the following prohibited argument(s) in 'args#{method}': "
                    f"{found_keys}. Model and datasets should be defined within the 'system'."
                )
        return fields


class ConfigSchema(BaseModel):
    _requires_: Optional = None
    project: Optional[str] = None
    vars: Dict[str, Any] = {}
    args: ArgsConfigSchema = ArgsConfigSchema()
    system: Dict[str, Any] = {}
    trainer: Dict[str, Any] = {}


# ----- LighterSystem schema -----


class DatasetSchema(BaseModel, SubscriptableModel):
    train: Optional[Dataset] = None
    val: Optional[Dataset] = None
    test: Optional[Dataset] = None
    predict: Optional[Dataset] = None


class SamplerSchema(BaseModel, SubscriptableModel):
    train: Optional[Sampler] = None
    val: Optional[Sampler] = None
    test: Optional[Sampler] = None
    predict: Optional[Sampler] = None


class CollateFnSchema(BaseModel, SubscriptableModel):
    train: Optional[Callable] = None
    val: Optional[Callable] = None
    test: Optional[Callable] = None
    predict: Optional[Callable] = None


class MetricsSchema(BaseModel, SubscriptableModel):
    train: Optional[Union[Any, List[Any], Dict[str, Any]]] = None
    val: Optional[Union[Any, List[Any], Dict[str, Any]]] = None
    test: Optional[Union[Any, List[Any], Dict[str, Any]]] = None

    @root_validator(pre=True)
    def setup_metrics(cls, fields):  # pylint: disable=no-self-argument
        for mode, metric in fields.items():
            if metric is not None:
                fields[mode] = MetricCollection(metric)
        return fields


class InputTargetPredSchema(BaseModel, SubscriptableModel):
    input: Optional[Union[Callable, List[Callable]]] = None
    target: Optional[Union[Callable, List[Callable]]] = None
    pred: Optional[Union[Callable, List[Callable]]] = None


class PostprocessingSchema(BaseModel, SubscriptableModel):
    batch: Dict[str, Optional[Union[Callable, List[Callable]]]] = Field(default_factory=dict)
    criterion: InputTargetPredSchema = Field(default_factory=InputTargetPredSchema)
    metrics: InputTargetPredSchema = Field(default_factory=InputTargetPredSchema)
    logging: InputTargetPredSchema = Field(default_factory=InputTargetPredSchema)
