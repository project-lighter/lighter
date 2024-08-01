from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from functools import partial

import pytorch_lightning as pl
import torch
from loguru import logger
from torch.nn import Module, ModuleDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Sampler
from torchmetrics import Metric, MetricCollection

from lighter.utils.collate import collate_replace_corrupted
from lighter.utils.misc import apply_fns, ensure_dict_schema, get_optimizer_stats, hasarg


class LighterSystem(pl.LightningModule):
    """_summary_

    Args:
        model (Module): Model.
        batch_size (int): Batch size.
        drop_last_batch (bool, optional): Whether the last batch in the dataloader should be dropped. Defaults to False.
        num_workers (int, optional): Number of dataloader workers. Defaults to 0.
        pin_memory (bool, optional): Whether to pin the dataloaders memory. Defaults to True.
        optimizer (Optimizer, optional): Optimizers. Defaults to None.
        scheduler (LRScheduler, optional): Learning rate scheduler. Defaults to None.
        criterion (Callable, optional): Criterion/loss function. Defaults to None.
        datasets (Dict[str, Dataset], optional): Datasets for train, val, test, and predict. Defaults to None.
        samplers (Dict[str, Sampler], optional): Samplers for train, val, test, and predict. Defaults to None.
        collate_fns (Dict[str, Union[Callable, List[Callable]]], optional):
            Collate functions for train, val, test, and predict. Defaults to None.
        metrics (Dict[str, Union[Metric, List[Metric], Dict[str, Metric]]], optional):
            Metrics for train, val, and test. Supports a single metric or a list/dict of `torchmetrics` metrics.
            Defaults to None.
        postprocessing (Dict[str, Union[Callable, List[Callable]]], optional):
            Functions to apply to:
                1) The batch returned from the train/val/test/predict Dataset. Defined separately for each.
                2) The input, target, or pred data prior to criterion/metrics/logging. Defined separately for each.

            Follow this structure (all keys are optional):
            ```
                batch:
                    train:
                    val:
                    test:
                    predict:
                criterion / metrics / logging:
                    input:
                    target:
                    pred:
            ```
            Note that the postprocessing of a latter stage stacks on top of the prior ones - for example,
            the logging postprocessing will be done on the data that has been postprocessed for the criterion
            and metrics earlier. Defaults to None.
        inferer (Callable, optional): Inferer must be a class with a `__call__` method that accepts two
            arguments - the input to infer over, and the model itself. Used in 'val', 'test', and 'predict'
            mode, but not in 'train'. Typically, an inferer is a sliding window or a patch-based inferer
            that will infer over the smaller parts of the input, combine them, and return a single output.
            The inferers provided by MONAI cover most of such cases (https://docs.monai.io/en/stable/inferers.html).
            Defaults to None.
    """

    def __init__(
        self,
        model: Module,
        batch_size: int,
        drop_last_batch: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        criterion: Optional[Callable] = None,
        datasets: Dict[str, Dataset] = None,
        samplers: Dict[str, Sampler] = None,
        collate_fns: Dict[str, Union[Callable, List[Callable]]] = None,
        metrics: Dict[str, Union[Metric, List[Metric], Dict[str, Metric]]] = None,
        postprocessing: Dict[str, Union[Callable, List[Callable]]] = None,
        inferer: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        # Bypass LightningModule's check for default methods. We define them in self.setup().
        self._init_placeholders_for_dataloader_and_step_methods()

        # Model setup
        self.model = model
        self.batch_size = batch_size

        # Criterion, optimizer, and scheduler
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # DataLoader specifics
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last_batch = drop_last_batch

        # Datasets, samplers, and collate functions
        self.datasets = self._init_datasets(datasets)
        self.samplers = self._init_samplers(samplers)
        self.collate_fns = self._init_collate_fns(collate_fns)

        # Metrics
        self.metrics = self._init_metrics(metrics)

        # Postprocessing
        self.postprocessing = self._init_postprocessing(postprocessing)

        # Inferer for val, test, and predict
        self.inferer = inferer

        # Flag that indicates whether the LightningModule methods have been defined. Used in `self.setup()`.
        self._lightning_module_methods_defined = False

    def forward(self, input: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor], Dict[str, torch.Tensor]]) -> Any:
        """Forward pass. Multi-input models are supported.

        Args:
            input (torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor], Dict[str, torch.Tensor]): Input to the model.

        Returns:
            Output of the model.
        """

        # Keyword arguments to pass to the forward method
        kwargs = {}
        # Add `epoch` argument if forward accepts it
        if hasarg(self.model.forward, "epoch"):
            kwargs["epoch"] = self.current_epoch
        # Add `step` argument if forward accepts it
        if hasarg(self.model.forward, "step"):
            kwargs["step"] = self.global_step

        return self.model(input, **kwargs)

    def _base_step(self, batch: Dict, batch_idx: int, mode: str) -> Union[Dict[str, Any], Any]:
        """Base step for all modes.

        Args:
            batch (Dict): Batch data as a containing "input", and optionally "target" and "id".
            batch_idx (int): Batch index. PyTorch Lightning requires it, even though it is not used here.
            mode (str): Operating mode. (train/val/test/predict)

        Returns:
            For the predict step, it returns pred only.
            For the training, validation, and test steps, it returns a dictionary
            containing loss, metrics, input, target, pred, and id. Loss is `None`
            for the test step. Metrics is `None` if no metrics are specified.
        """
        # Allow postprocessing on batch data. Can be used to restructure the batch data into the required format.
        batch = apply_fns(batch, self.postprocessing["batch"][mode])

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

        # Unpack the batch. The target and id are optional. If not provided, they are set to None for internal use.
        input = batch["input"]
        target = batch.get("target", None)
        id = batch.get("id", None)

        # Forward
        if self.inferer and mode in ["val", "test", "predict"]:
            pred = self.inferer(input, self)
        else:
            pred = self(input)

        # Postprocessing for loss calculation.
        input = apply_fns(input, self.postprocessing["criterion"]["input"])
        target = apply_fns(target, self.postprocessing["criterion"]["target"])
        pred = apply_fns(pred, self.postprocessing["criterion"]["pred"])

        # Predict mode stops here.
        if mode == "predict":
            # Postprocessing for logging/writing.
            pred = apply_fns(pred, self.postprocessing["logging"]["pred"])
            return {"pred": pred, "id": id}

        # Calculate the loss.
        loss = None
        if mode in ["train", "val"]:
            # When target is not provided, pass only the predictions to the criterion.
            loss = self.criterion(pred) if target is None else self.criterion(pred, target)

        # Postprocessing for metrics.
        input = apply_fns(input, self.postprocessing["metrics"]["input"])
        target = apply_fns(target, self.postprocessing["metrics"]["target"])
        pred = apply_fns(pred, self.postprocessing["metrics"]["pred"])

        # Calculate the step metrics. # TODO: Remove the "_" prefix when fixed https://github.com/pytorch/pytorch/issues/71203
        metrics = self.metrics["_" + mode](pred, target) if self.metrics["_" + mode] is not None else None

        # Postprocessing for logging/writing.
        input = apply_fns(input, self.postprocessing["logging"]["input"])
        target = apply_fns(target, self.postprocessing["logging"]["target"])
        pred = apply_fns(pred, self.postprocessing["logging"]["pred"])

        # Ensure that a dict of losses has a 'total' key.
        if isinstance(loss, dict) and "total" not in loss:
            raise ValueError(
                "The loss dictionary must include a 'total' key that combines all sublosses. "
                "Example: {'total': combined_loss, 'subloss1': loss1, ...}"
            )

        # Logging
        self._log_stats(loss, metrics, mode, batch_idx)

        # Return the loss as required by Lightning as well as other data that can be used in hooks or callbacks.
        return {
            "loss": loss["total"] if isinstance(loss, dict) else loss,
            "metrics": metrics,
            "input": input,
            "target": target,
            "pred": pred,
            "id": id,
        }

    def _log_stats(
        self, loss: Union[torch.Tensor, Dict[str, torch.Tensor]], metrics: MetricCollection, mode: str, batch_idx: int
    ) -> None:
        """
        Logs the loss, metrics, and optimizer statistics.
        Args:
            loss (Union[torch.Tensor, Dict[str, torch.Tensor]]): Calculated loss or a dict of sublosses.
            metrics (MetricCollection): Calculated metrics.
            mode (str): Mode of operation (train/val/test/predict).
            batch_idx (int): Index of current batch.
        """
        if self.trainer.logger is None:
            return

        on_step_log = partial(self.log, logger=True, batch_size=self.batch_size, on_step=True, on_epoch=False, sync_dist=False)
        on_epoch_log = partial(self.log, logger=True, batch_size=self.batch_size, on_step=False, on_epoch=True, sync_dist=True)

        # Loss
        if loss is not None:
            if not isinstance(loss, dict):
                on_step_log(f"{mode}/loss/step", loss)
                on_epoch_log(f"{mode}/loss/epoch", loss)
            else:
                for name, subloss in loss.items():
                    on_step_log(f"{mode}/loss/{name}/step", subloss)
                    on_epoch_log(f"{mode}/loss/{name}/epoch", subloss)
        # Metrics
        if metrics is not None:
            for name, metric in metrics.items():
                on_step_log(f"{mode}/metrics/{name}/step", metric)
                on_epoch_log(f"{mode}/metrics/{name}/epoch", metric)
        # Optimizer's lr, momentum, beta. Logged in train mode and once per epoch.
        if mode == "train" and batch_idx == 0:
            for name, optimizer_stat in get_optimizer_stats(self.optimizer).items():
                on_epoch_log(f"{mode}/{name}", optimizer_stat)

    def _base_dataloader(self, mode: str) -> DataLoader:
        """Instantiate the dataloader for a mode (train/val/test/predict).
        Includes a collate function that enables the DataLoader to replace
        None's (alias for corrupted examples) in the batch with valid examples.
        To make use of it, write a try-except in your Dataset that handles
        corrupted data by returning None instead.

        Args:
            mode (str): Mode of operation for which to create the dataloader ["train", "val", "test", "predict"].

        Returns:
            Instantiated DataLoader.
        """
        dataset = self.datasets[mode]
        sampler = self.samplers[mode]
        collate_fn = self.collate_fns[mode]

        if dataset is None:
            raise ValueError(f"Please specify '{mode}' dataset in the 'datasets' key of the config.")

        # Batch size is 1 when using an inference for two reasons:
        # 1) Inferer separates an input into multiple parts, forming a batch of its own.
        # 2) The val/test/pred data usually differ in their shape, so they cannot be stacked into a batch.
        batch_size = self.batch_size
        if self.inferer is not None and mode in ["val", "test", "predict"]:
            logger.info(f"Setting the '{mode}' mode dataloader to batch size of 1 because an inferer is provided.")
            batch_size = 1

        # A dataset can return None when a corrupted example occurs. This collate
        # function replaces None's with valid examples from the dataset.
        collate_fn = partial(collate_replace_corrupted, dataset=dataset, default_collate_fn=collate_fn)
        return DataLoader(
            dataset,
            sampler=sampler,
            shuffle=(mode == "train" and sampler is None),
            batch_size=batch_size,
            drop_last=(self.drop_last_batch if mode == "train" else False),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def configure_optimizers(self) -> Dict:
        """LightningModule method. Returns optimizers and, if defined, schedulers.

        Returns:
            Optimizer and, if defined, scheduler.
        """
        if self.optimizer is None:
            raise ValueError("Please specify 'system.optimizer' in the config.")
        if self.scheduler is None:
            return {"optimizer": self.optimizer}
        else:
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def setup(self, stage: str) -> None:
        """Automatically called by the LightningModule after the initialization.
        `LighterSystem`'s setup checks if the required dataset is provided in the config and
        sets up LightningModule methods for the stage in which the system is.

        Args:
            stage (str): Passed automatically by PyTorch Lightning. ["fit", "validate", "test"].
        """
        # Stage-specific PyTorch Lightning methods. Defined dynamically so that the system
        # only has methods used in the stage and for which the configuration was provided.
        if not self._lightning_module_methods_defined:
            del (
                self.train_dataloader,
                self.training_step,
                self.val_dataloader,
                self.validation_step,
                self.test_dataloader,
                self.test_step,
                self.predict_dataloader,
                self.predict_step,
            )
            # Prevents the methods from being defined again. This is needed because `Trainer.tune()`
            # calls the `self.setup()` method whenever it runs for a new parameter.
            self._lightning_module_methods_defined = True

        # Training methods.
        if stage in ["fit", "tune"]:
            self.train_dataloader = partial(self._base_dataloader, mode="train")
            self.training_step = partial(self._base_step, mode="train")

        # Validation methods. Required in 'validate' stage and optionally in 'fit' or 'tune' stage.
        if stage == "validate" or (stage in ["fit", "tune"] and self.datasets["val"] is not None):
            self.val_dataloader = partial(self._base_dataloader, mode="val")
            self.validation_step = partial(self._base_step, mode="val")

        # Test methods.
        if stage == "test":
            self.test_dataloader = partial(self._base_dataloader, mode="test")
            self.test_step = partial(self._base_step, mode="test")

        # Predict methods.
        if stage == "predict":
            self.predict_dataloader = partial(self._base_dataloader, mode="predict")
            self.predict_step = partial(self._base_step, mode="predict")

    @property
    def learning_rate(self) -> float:
        """Get the learning rate of the optimizer. Ensures compatibility with the Tuner's 'lr_find()' method."""
        if len(self.optimizer.param_groups) > 1:
            raise ValueError("The learning rate is not available when there are multiple optimizer parameter groups.")
        return self.optimizer.param_groups[0]["lr"]

    @learning_rate.setter
    def learning_rate(self, value) -> None:
        """Set the learning rate of the optimizer. Ensures compatibility with the Tuner's 'lr_find()' method."""
        if len(self.optimizer.param_groups) > 1:
            raise ValueError("The learning rate is not available when there are multiple optimizer parameter groups.")
        self.optimizer.param_groups[0]["lr"] = value

    def _init_placeholders_for_dataloader_and_step_methods(self) -> None:
        """
        Initializes placeholders for dataloader and step methods.

        `LighterSystem` dynamically defines the `..._dataloader()`and `..._step()` methods
        in the `self.setup()` method. However, when `LightningModule` excepts them to be defined
        at init. To prevent it from throwing an error, the `..._dataloader()` and `..._step()`
        are initially defined as `lambda: None`, before `self.setup()` is called.
        """
        self.train_dataloader = self.training_step = lambda: None
        self.val_dataloader = self.validation_step = lambda: None
        self.test_dataloader = self.test_step = lambda: None
        self.predict_dataloader = self.predict_step = lambda: None

    def _init_datasets(self, datasets: Dict[str, Optional[Dataset]]):
        """Ensures that the datasets have the predefined schema."""
        return ensure_dict_schema(datasets, {"train": None, "val": None, "test": None, "predict": None})

    def _init_samplers(self, samplers: Dict[str, Optional[Sampler]]):
        """Ensures that the samplers have the predefined schema"""
        return ensure_dict_schema(samplers, {"train": None, "val": None, "test": None, "predict": None})

    def _init_collate_fns(self, collate_fns: Dict[str, Optional[Callable]]):
        """Ensures that the collate functions have the predefined schema."""
        return ensure_dict_schema(collate_fns, {"train": None, "val": None, "test": None, "predict": None})

    def _init_metrics(self, metrics: Dict[str, Optional[Union[Metric, List[Metric], Dict[str, Metric]]]]):
        """Ensures that the metrics have the predefined schema. Wraps each mode's metrics in
        a MetricCollection, and finally registers them with PyTorch using a ModuleDict.
        """
        metrics = ensure_dict_schema(metrics, {"train": None, "val": None, "test": None})
        for mode, metric in metrics.items():
            metrics[mode] = MetricCollection(metric) if metric is not None else None
        # TODO: Remove the prefix addition line below when fixed https://github.com/pytorch/pytorch/issues/71203
        metrics = {f"_{k}": v for k, v in metrics.items()}
        return ModuleDict(metrics)

    def _init_postprocessing(self, postprocessing: Dict[str, Optional[Union[Callable, List[Callable]]]]):
        """Ensures that the postprocessing functions have the predefined schema."""
        mode_subschema = {"train": None, "val": None, "test": None, "predict": None}
        data_subschema = {"input": None, "target": None, "pred": None}
        schema = {"batch": mode_subschema, "criterion": data_subschema, "metrics": data_subschema, "logging": data_subschema}
        return ensure_dict_schema(postprocessing, schema)
