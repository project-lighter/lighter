from __future__ import annotations
from abc import ABC
from enum import StrEnum
from typing import Any, Callable, Dict, List, Optional, Union, Protocol, Sequence

from functools import partial, partialmethod

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.data import extract_batch_size
from torch.nn import Module, ModuleDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric, MetricCollection

from lighter.defs import ModeEnum
from lighter.metrics.metrics import MetricContainerCollection, MetricContainer, MetricResultDims
from lighter.postprocessing.pipeline import ProcessingPipelineDefinition, ProcessingPipeline
from lighter.utils.misc import ensure_dict_schema, get_optimizer_stats, hasarg


class ModelAdapter(Protocol):
    def __call__(self, model: Module, batch: Dict[str, Any]) -> Dict[str, Any]:
        pass

class DefaultModelAdapter(ModelAdapter):
    def __call__(self, model: Module, batch: Dict[str, Any]) -> Dict[str, Any]:
        return model(batch["input"])

class CriterionAdaptor(Protocol):
    def __call__(self, criterion: Callable, batch: Dict[str, Any]) -> Dict[str, torch.Tensor] | torch.Tensor:
        pass

class DefaultCriterionAdapter(CriterionAdaptor):
    def __call__(self, criterion: Callable, batch: Dict[str, Any]) -> Dict[str, torch.Tensor] | torch.Tensor:
        return criterion(batch["pred"], batch["target"]) if "target" in batch else criterion(batch["pred"])


class DataLogger(Protocol):
    def log_step(self, system: LighterSystem, data: Dict[str, Any], mode: ModeEnum, batch_idx: int | None = None, dataloader_idx: int | None = None) -> None:
        pass
    def log_epoch(self, system: LighterSystem, metrics: Dict[MetricContainer, torch.Tensor | None], mode: ModeEnum) -> None:
        pass



class LogStageEnum(StrEnum):
    STEP = "step"
    EPOCH = "epoch"
class BaseMetricLogger(DataLogger):

    def _get_metric_name(self,metric: MetricContainer) -> str:
        return metric.name

    def _get_log_key(self, mode: ModeEnum, metric: MetricContainer, log_stage: LogStageEnum,class_name:str=None) -> str:
        key = f"{mode}/{log_stage}/{self._get_metric_name(metric)}"
        if class_name:
            key = f"{key}/{class_name}"
        return key


    def check_and_log(self, system: LighterSystem, key: str, value: torch.Tensor,*, on_step: bool, on_epoch:bool, logged_keys: set) -> None:
        if key in logged_keys:
            raise ValueError(f"Key '{key}' has already been logged. Please ensure that each key is logged only once.")
        logged_keys.add(key)
        system.log(key, value, on_step=on_step, on_epoch=on_epoch, sync_dist=False)

    def log_step(self, system: LighterSystem, data: Dict[str, Any], mode: ModeEnum, batch_idx: int | None = None, dataloader_idx: int | None = None) -> None:
        logged_keys = set()
        losses = data.get("losses", None)
        if losses is None and "loss" in data:
            losses = {"loss": data["loss"]}
        if losses:
            for loss_name, loss_value in losses.items():
                postfix = f"losses/{loss_name}" if loss_name != "loss" else "loss"
                key = f"{mode}/{LogStageEnum.STEP}/{postfix}"
                self.check_and_log(system, key, loss_value, on_step=True, on_epoch=False, logged_keys=logged_keys)
                key = f"{mode}/{LogStageEnum.EPOCH}/{postfix}"
                self.check_and_log(system, key, loss_value, on_step=False, on_epoch=True, logged_keys=logged_keys)
        metrics: Dict[MetricContainer, torch.Tensor | None]
        if metrics := data.get("metrics", None): # type: ignore
            for metric, value in metrics.items():
                if value is not None:
                    batch_dim_idx = metric.step_dims.index(MetricResultDims.BATCH)
                    if batch_dim_idx != -1:
                        value = torch.mean(value, dim=batch_dim_idx)
                    if MetricResultDims.CLASS in metric.step_dims:
                        assert value.dim() == 1
                        for class_name, class_value in zip(metric.class_names, value.unbind(dim=0)):
                            key = self._get_log_key(mode, metric, LogStageEnum.STEP,class_name)
                            self.check_and_log(system, key, class_value, on_step=True, on_epoch=False, logged_keys=logged_keys)
                    else:
                        key = self._get_log_key(mode, metric, LogStageEnum.STEP)
                        self.check_and_log(system, key, value, on_step=True, on_epoch=False, logged_keys=logged_keys)
    def log_epoch(self, system: LighterSystem, metrics: Dict[MetricContainer, torch.Tensor | None], mode: ModeEnum) -> None:
        logged_keys = set()
        for metric, value in metrics.items():
            if value is not None:
                batch_dim_idx = metric.step_dims.index(MetricResultDims.BATCH)
                if batch_dim_idx != -1:
                    value = torch.mean(value, dim=batch_dim_idx)
                if MetricResultDims.CLASS in metric.step_dims:
                    assert value.dim() == 1
                    for class_name, class_value in zip(metric.class_names, value.unbind(dim=0)):
                        key = self._get_log_key(mode, metric, LogStageEnum.EPOCH,class_name)
                        self.check_and_log(system, key, class_value, on_step=False, on_epoch=True, logged_keys=logged_keys)
                else:
                    key = self._get_log_key(mode, metric, LogStageEnum.EPOCH)
                    self.check_and_log(system, key, value, on_step=False, on_epoch=True, logged_keys=logged_keys)

class LighterSystem(pl.LightningModule):
    """_summary_

    Args:
        model (Module): The model.
        batch_size (int): Batch size.
        drop_last_batch (bool, optional): Whether the last batch in the dataloader should be dropped. Defaults to False.
        num_workers (int, optional): Number of dataloader workers. Defaults to 0.
        pin_memory (bool, optional): Whether to pin the dataloaders memory. Defaults to True.
        optimizer (Optimizer, optional): Optimizers. Defaults to None.
        scheduler (LRScheduler, optional): Learning rate scheduler. Defaults to None.
        criterion (Callable, optional): Criterion/loss function. Defaults to None.
        datasets (Dict[str, Dataset], optional): Datasets for train, val, test, and predict. Defaults to None.
        dataloaders: Dataloaders for train, val, test, and predict. Defaults to None.
        samplers (Dict[str, Sampler], optional): Samplers for train, val, test, and predict. Defaults to None.
        collate_fns (Dict[str, Union[Callable, List[Callable]]], optional):
            Collate functions for train, val, test, and predict. Defaults to None.
        metrics (Dict[str, Union[Metric, List[Metric], Dict[str, Metric]]], optional):
            Metrics for train, val, and test. Supports a single metric or a list/dict of `torchmetrics` metrics.
            Defaults to None.
        postprocessing (ProcessingPipelineDefinition | dict, optional):
            Postprocessingpipeline
        inferer (Callable, optional): The inferer must be a class with a `__call__` method that accepts two
            arguments - the input to infer over, and the model itself. Used in 'val', 'test', and 'predict'
            mode, but not in 'train'. Typically, an inferer is a sliding window or a patch-based inferer
            that will infer over the smaller parts of the input, combine them, and return a single output.
            The inferers provided by MONAI cover most of such cases (https://docs.monai.io/en/stable/inferers.html).
            Defaults to None.
    """
    metrics: MetricContainerCollection
    def __init__(
        self,
        model: Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        criterion: Optional[Callable] = None,
        criterion_adapter: Optional[CriterionAdaptor] = None,
        datasets: Dict[str, Dataset] = None,
        dataloaders: Dict[str, Callable] = None,
        metrics: Dict[str, Union[Metric, List[Metric], Dict[str, Metric]]] = None,
        postprocessing: ProcessingPipelineDefinition | dict = None,
        inferer: Optional[Callable] = None,
        model_adapter: Optional[ModelAdapter] = None,
        data_logger: Optional[DataLogger] = None,
    ) -> None:
        super().__init__()

        # Model setup
        self.model = model
        self.model_adapter = model_adapter or DefaultModelAdapter()

        # Criterion, optimizer, and scheduler
        self.criterion = criterion
        self.criterion_adapter = criterion_adapter or DefaultCriterionAdapter()
        self.optimizer = optimizer
        self.scheduler = scheduler

        # DataLoader specifics
        self.dataloaders = dataloaders or {}

        # Datasets, samplers, and collate functions
        self.datasets = self._init_datasets(datasets)

        # Metrics
        self.metrics = self._init_metrics(metrics)

        # Postprocessing
        self.postprocessing = self._init_postprocessing(postprocessing)

        # Inferer for val, test, and predict
        self.inferer = inferer

        # This is needed to remove the validation step/dataloader if the validation dataset is not provided.
        if self.datasets.get(ModeEnum.VAL, None) is None:
            self.validation_step = None
            self.val_dataloader = None
        self.data_logger = data_logger or BaseMetricLogger()

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass. Multi-input models are supported.

        Args:
            input (torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor], Dict[str, torch.Tensor]): input to the model.

        Returns:
            Any: output of the model.
        """

        # TODO This seems quite use-case specific, maybe we can add this to a custom adapter?
        # Add `epoch` argument if forward accepts it
        if hasarg(self.model.forward, "epoch"):
            kwargs["epoch"] = self.current_epoch
        # Add `step` argument if forward accepts it
        if hasarg(self.model.forward, "step"):
            kwargs["step"] = self.global_step

        return self.model(*args, **kwargs)


    def calculate_loss(self, result: Dict[str, Any], mode: ModeEnum):
        """Calculate the loss from the result dictionary.

        Args:
            result (Dict[str, Any]): Dictionary containing the loss.
                This will be modified to include the loss value.
            mode (str): The operating mode.
        """
        if mode in ["train", "val"]:
            # When target is not provided, pass only the predictions to the criterion.
            loss = self.criterion_adapter(self.criterion, result)
            if isinstance(loss, dict):
                if "loss" not in result:
                    raise ValueError("Criterion must return a dict of tensors."
                                     " It must contain a 'loss' key, which will be used as the optimization target."
                                     " Other keys can be used for logging purposes (e.g. CE+Dice loss).")
                result["losses"] = loss
                result["loss"] = loss["loss"]
            elif isinstance(loss, torch.Tensor):
                result["loss"] = loss
                result["losses"] = {"loss": loss}
            else:
                raise ValueError("Criterion must return a tensor or a dict of tensors.")

    def _base_step(self, batch: Dict, batch_idx: int, mode: ModeEnum) -> Union[Dict[str, Any], Any]:
        """Base step for all modes.

        Args:
            batch (Dict): The batch data as a containing "input", and optionally "target" and "id".
            batch_idx (int): Batch index. Not used, but PyTorch Lightning requires it.
            mode (str): The operating mode. (train/val/test/predict)

        Returns:
            Union[Dict[str, Any], Any]: For the predict step, it returns pred only.
                For the training, validation, and test steps, it returns a dictionary
                containing loss, metrics, input, target, pred, and id. Loss is `None`
                for the test step. Metrics is `None` if no metrics are specified.
        """
        # Allow postprocessing on batch data. Can be used to restructure the batch data into the required format.
        # batch = apply_fns(batch, self.postprocessing["batch"][mode])

        # Verify that the batch is a dict with valid keys.
        batch_err_msg = "Batch must be a dict with keys:\n\t- 'input'\n\t- 'target' (optional)\n\t- 'id' (optional)\n"
        if not isinstance(batch, dict):
            raise TypeError(batch_err_msg + f"Batch type found: '{type(batch).__name__}'.")
        if set(batch.keys()) not in [{"input"}, {"input", "target"}, {"input", "id"}, {"input", "target", "id"}]:
            raise ValueError(batch_err_msg + f"Batch keys found: {batch.keys()}.")

        # Ensure that the returned values of the optional batch keys are not None.
        for key in ["target", "id"]:
            if key in batch and batch[key] is None:
                raise ValueError(f"Batch's '{key}' value cannot be None. If '{key}' should not exist, omit it.")

        # Forward
        if self.inferer and mode in ["val", "test", "predict"]:
            raise NotImplementedError("Inferer is not yet implemented.")
        else:
            pred = self.model_adapter(self, batch)
        result = {
            **batch,
            "pred": pred,
        }
        pipeline_instance = ProcessingPipeline(self.postprocessing, dict(result))
        result["pipeline"] = pipeline_instance
        # Predict mode stops here.
        if mode == "predict":
            # Postprocessing for logging/writing.
            return result

        # Calculate the loss.
        self.calculate_loss(result, mode)

        metrics = self.metrics(result, mode) if self.metrics is not None else None
        if metrics is not None:
            result["metrics"] = metrics
        # Logging
        self.data_logger.log_step(self, result, mode, batch_idx=batch_idx)

        return result

    training_step = partialmethod(_base_step, mode="train")
    validation_step = partialmethod(_base_step, mode="val")
    test_step = partialmethod(_base_step, mode="test")
    predict_step = partialmethod(_base_step, mode="predict")


    def _base_dataloader(self, mode: str) -> DataLoader:
        """Instantiate the dataloader for a mode (train/val/test/predict).
        Includes a collate function that enables the DataLoader to replace
        None's (alias for corrupted examples) in the batch with valid examples.
        To make use of it, write a try-except in your Dataset that handles
        corrupted data by returning None instead.

        Args:
            mode (str): Mode of operation for which to create the dataloader ["train", "val", "test", "predict"].

        Returns:
            DataLoader: Instantiated DataLoader.
        """



        dataset = self.datasets[mode]

        if dataset is None:
            raise ValueError(f"Please specify '{mode}' dataset in the 'datasets' key of the config.")

        if mode in self.dataloaders:
            return self.dataloaders[mode](dataset)


        return DataLoader(
            dataset,
            batch_size=1,
            drop_last=(self.drop_last_batch if mode == "train" else False),
        )
    train_dataloader = partialmethod(_base_dataloader, mode="train")
    val_dataloader = partialmethod(_base_dataloader, mode="val")
    test_dataloader = partialmethod(_base_dataloader, mode="test")
    predict_dataloader = partialmethod(_base_dataloader, mode="predict")
    def configure_optimizers(self) -> Dict:
        """LightningModule method. Returns optimizers and, if defined, schedulers.

        Returns:
            Dict: optimizer and, if defined, scheduler.
        """
        if self.optimizer is None:
            raise ValueError("Please specify 'system.optimizer' in the config.")
        if self.scheduler is None:
            return {"optimizer": self.optimizer}
        else:
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}



    def _base_epoch_start(self, mode: ModeEnum):
        if self.metrics is not None:
            self.metrics.reset(mode)

    on_train_epoch_start = partialmethod(_base_epoch_start, mode=ModeEnum.TRAIN)
    on_validation_epoch_start = partialmethod(_base_epoch_start, mode=ModeEnum.VAL)
    on_test_epoch_start = partialmethod(_base_epoch_start, mode=ModeEnum.TEST)
    on_predict_epoch_start = partialmethod(_base_epoch_start, mode=ModeEnum.PREDICT)

    def _base_epoch_end(self, mode: ModeEnum):
        if self.metrics is not None:
            if metrics := self.metrics.compute(mode):
                self.data_logger.log_epoch(self, metrics, mode)

    on_train_epoch_end = partialmethod(_base_epoch_end, mode=ModeEnum.TRAIN)
    on_validation_epoch_end = partialmethod(_base_epoch_end, mode=ModeEnum.VAL)
    on_test_epoch_end = partialmethod(_base_epoch_end, mode=ModeEnum.TEST)
    on_predict_epoch_end = partialmethod(_base_epoch_end, mode=ModeEnum.PREDICT)


    def _init_datasets(self, datasets: Dict[str, Optional[Dataset]]):
        """Ensures that the datasets have the predefined schema."""
        return ensure_dict_schema(datasets, {"train": None, "val": None, "test": None, "predict": None})


    def _init_metrics(self, metrics: Sequence[dict[str,Any]]) -> MetricContainerCollection:
        instantiated_metrics = []
        for metric in metrics:
            try:
                if isinstance(metric, dict):
                    instantiated_metrics.append(MetricContainer(**metric))
                else:
                    raise ValueError(f"Invalid metric definition: {metric}")
            except Exception as e:
                raise ValueError(f"Error while instantiating metric: {metric}") from e
        return MetricContainerCollection(instantiated_metrics)

    def _init_postprocessing(self, postprocessing: ProcessingPipelineDefinition | dict) -> ProcessingPipelineDefinition:
        """Ensures that the postprocessing functions have the predefined schema."""
        if postprocessing is None:
            return ProcessingPipelineDefinition({})
        if isinstance(postprocessing, ProcessingPipelineDefinition):
            return postprocessing
        else:
            return ProcessingPipelineDefinition(postprocessing)

