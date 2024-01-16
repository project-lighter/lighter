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
from lighter.utils.misc import apply_fns, ensure_dict_schema, get_name, get_optimizer_stats, hasarg


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
        samplers (Dict[str, Sampler], optional): Samplers for train, val, test, and predict. Defaults to None.
        collate_fns (Dict[str, Union[Callable, List[Callable]]], optional):
            Collate functions for train, val, test, and predict. Defaults to None.
        metrics (Dict[str, Union[Metric, List[Metric], Dict[str, Metric]]], optional):
            Metrics for train, val, and test. Supports a single metric or a list/dict of `torchmetrics` metrics.
            Defaults to None.
        postprocessing (Dict[str, Union[Callable, List[Callable]]], optional):
            Postprocessing functions for input, target, and pred, for three stages - criterion, metrics,
            and logging. The postprocessing is done before each stage - for example, criterion postprocessing
            will be done prior to loss calculation. Note that the postprocessing of a latter stage stacks on
            top of the previous one(s) - for example, the logging postprocessing will be done on the data that
            has been postprocessed for the criterion and metrics earlier. Defaults to None.
        inferer (Callable, optional): The inferer must be a class with a `__call__` method that accepts two
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
        self.drop_last_batch = drop_last_batch

        # Criterion, optimizer, and scheduler
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # DataLoader specifics
        self.num_workers = num_workers
        self.pin_memory = pin_memory

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

        # Checks
        self._lightning_module_methods_defined = False
        self._target_not_used_reported = False
        self._batch_type_reported = False

    def forward(self, input: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor], Dict[str, torch.Tensor]]) -> Any:
        """Forward pass. Multi-input models are supported.

        Args:
            input (torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor], Dict[str, torch.Tensor]): input to the model.

        Returns:
            Any: output of the model.
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

    def _base_step(self, batch: Union[List, Tuple], batch_idx: int, mode: str) -> Union[Dict[str, Any], Any]:
        """Base step for all modes ("train", "val", "test", "predict")

        Args:
            batch (List, Tuple):
                output of the DataLoader and input to the model.
            batch_idx (int): index of the batch. Not used, but PyTorch Lightning requires it.
            mode (str): mode in which the system is.

        Returns:
            Union[Dict[str, Any], Any]: For the training, validation and test step, it returns
                a dict containing loss, metrics, input, target, and pred. Loss will be `None`
                for the test step. Metrics will be `None` if no metrics are specified.

                For predict step, it returns pred only.
        """
        # Batch type check:
        # - Dict: must contain "input" and "target" keys, and optionally "id" key.
        if isinstance(batch, dict):
            if set(batch.keys()) not in [{"input", "target"}, {"input", "target", "id"}]:
                raise ValueError(
                    "A batch dict must have 'input', 'target', and, "
                    f"optionally 'id', as keys, but found {list(batch.keys())}"
                )
            batch["id"] = None if "id" not in batch else batch["id"]
        # - List/tuple: must contain two elements - input and target. After the check, convert it to dict.
        elif isinstance(batch, (list, tuple)):
            if len(batch) != 2:
                raise ValueError(
                    f"A batch must consist of 2 elements - input and target. However, {len(batch)} "
                    "elements were found. Note: if target does not exist, return `None` as target."
                )
            batch = {"input": batch[0], "target": batch[1], "id": None}
        # - Other types are not allowed.
        else:
            raise TypeError(
                "A batch must be a list, a tuple, or a dict."
                "A batch dict must have 'input' and 'target' keys, and optionally 'id'."
                "A batch list or a tuple must have 2 elements - input and target."
                "If target does not exist, return `None` as target."
            )

        # Split the batch into input, target, and id.
        input, target, id = batch["input"], batch["target"], batch["id"]

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
        loss = self._calculate_loss(pred, target) if mode in ["train", "val"] else None

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

        # Logging
        self._log_loss_metrics_and_optimizer(loss, metrics, self.optimizer, mode, batch_idx)

        return {"loss": loss, "metrics": metrics, "input": input, "target": target, "pred": pred, "id": id}

    def _calculate_loss(
        self, pred: Union[torch.Tensor, List, Tuple, Dict], target: Union[torch.Tensor, List, Tuple, Dict, None]
    ) -> torch.Tensor:
        """Calculates the loss.
        The method handles cases where the criterion function does not accept a `target` argument. If the criterion
        function does not accept a `target` argument, the LighterSystem passes only the predicted values to the criterion.

        Args:
            pred (torch.Tensor, List, Tuple, Dict): the predicted values from the model.
            target (torch.Tensor, List, Tuple, Dict, None): the target/label.

        Returns:
            torch.Tensor: the calculated loss.
        """

        # Keyword arguments to pass to the loss/criterion function
        kwargs = {}
        # Add `target` argument if forward accepts it.
        if hasarg(self.criterion.forward, "target"):
            kwargs["target"] = target
        else:
            if not self._target_not_used_reported and not self.trainer.sanity_checking:
                self._target_not_used_reported = True
                logger.info(
                    f"The criterion `{get_name(self.criterion, True)}` "
                    "has no `target` argument. In such cases, the LighterSystem "
                    "passes only the predicted values to the criterion. "
                    "If this is not the behavior you expected, redefine your "
                    "criterion so that it has a `target` argument."
                )
        return self.criterion(pred, **kwargs)

    def _log_loss_metrics_and_optimizer(
        self, loss: torch.Tensor, metrics: MetricCollection, optimizer: Optimizer, mode: str, batch_idx: int
    ) -> None:
        """
        Logs the loss, metrics, and optimizer statistics.

        Args:
            loss (torch.Tensor): The calculated loss.
            metrics (MetricCollection): The calculated metrics.
            optimizer (Optimizer): The optimizer used in the model.
            mode (str): The mode of operation (train/val/test/predict).
            batch_idx (int): The index of the current batch.
        """
        if self.trainer.logger is None:
            return

        default_kwargs = {"logger": True, "batch_size": self.batch_size}
        step_kwargs = {"on_epoch": False, "on_step": True}
        epoch_kwargs = {"on_epoch": True, "on_step": False}
        # Loss
        if loss is not None:
            self.log(f"{mode}/loss/step", loss, **default_kwargs, **step_kwargs)
            self.log(f"{mode}/loss/epoch", loss, **default_kwargs, **epoch_kwargs, sync_dist=True)
        # Metrics
        if metrics is not None:
            for k, v in metrics.items():
                self.log(f"{mode}/metrics/{k}/step", v, **default_kwargs, **step_kwargs)
                self.log(f"{mode}/metrics/{k}/epoch", v, **default_kwargs, **epoch_kwargs, sync_dist=True)
        # Optimizer's lr, momentum, beta. Logged in train mode and once per epoch.
        if mode == "train" and batch_idx == 0:
            for k, v in get_optimizer_stats(self.optimizer).items():
                self.log(f"{mode}/{k}", v, **default_kwargs, **epoch_kwargs)

    def _base_dataloader(self, mode: str) -> DataLoader:
        """Instantiate the dataloader for a mode (train/val/test/predict).
        Includes a collate function that enables the DataLoader to replace
        None's (alias for corrupted examples) in the batch with valid examples.
        To make use of it, write a try-except in your Dataset that handles
        corrupted data by returning None instead.

        Args:
            mode (str): mode for which to create the dataloader ["train", "val", "test", "predict"].

        Returns:
            DataLoader: instantiated DataLoader.
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
            Dict: optimizer and, if defined, scheduler.
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
            stage (str): passed by PyTorch Lightning. ["fit", "validate", "test"].
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
        subschema = {"input": None, "target": None, "pred": None}
        schema = {"criterion": subschema, "metrics": subschema, "logging": subschema}
        return ensure_dict_schema(postprocessing, schema)
