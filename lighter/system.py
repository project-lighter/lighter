from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sys
from functools import partial

import pytorch_lightning as pl
import torch
from loguru import logger
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Sampler
from torchmetrics import Metric, MetricCollection

from lighter.utils.collate import collate_replace_corrupted
from lighter.utils.misc import apply_fns, ensure_dict_schema, ensure_list, get_name, hasarg


class LighterSystem(pl.LightningModule):
    """_summary_

    Args:
        model (Module): the model.
        batch_size (int): batch size.
        drop_last_batch (bool, optional): whether the last batch in the dataloader
            should be dropped. Defaults to False.
        num_workers (int, optional): number of dataloader workers. Defaults to 0.
        pin_memory (bool, optional): whether to pin the dataloaders memory. Defaults to True.
        optimizer (Optional[Union[Optimizer, List[Optimizer]]], optional):
            a single or a list of optimizers. Defaults to None.
        scheduler (Optional[Union[Callable, List[Callable]]], optional):
            a single or a list of schedulers. Defaults to None.
        criterion (Optional[Callable], optional):
            criterion/loss function. Defaults to None.
        datasets (Optional[Dict[str, Optional[Dataset]]], optional):
            datasets for train, val, test, and predict. Supports Defaults to None.
        samplers (Optional[Dict[str, Optional[Sampler]]], optional):
            samplers for train, val, test, and predict. Defaults to None.
        collate_fns (Optional[Dict[str, Optional[Callable]]], optional):
            collate functions for train, val, test, and predict. Defaults to None.
        metrics (Optional[Dict[str, Optional[Union[Metric, List[Metric]]]]], optional):
            metrics for train, val, and test. Supports a single metric or a list of metrics,
            implemented using `torchmetrics`. Defaults to None.
        postprocessing (Optional[Dict[str, Optional[Callable]]], optional):
            Postprocessing functions for input, target, and pred, for three stages - criterion, metrics,
            and logging. The postprocessing is done before each stage - for example, criterion postprocessing
            will be done prior to loss calculation. Note that the postprocessing of a latter stage stacks on
            top of the previous one(s) - for example, the logging postprocessing will be done on the data that
            has been postprocessed for the criterion and metrics earlier. Defaults to None.
        inferer (Optional[Callable], optional): the inferer must be a class with a `__call__`
            method that accepts two arguments - the input to infer over, and the model itself.
            Used in 'val', 'test', and 'predict' mode, but not in 'train'. Typically, an inferer
            is a sliding window or a patch-based inferer that will infer over the smaller parts of
            the input, combine them, and return a single output. The inferers provided by MONAI
            cover most of such cases (https://docs.monai.io/en/stable/inferers.html). Defaults to None.
    """

    def __init__(
        self,
        model: Module,
        batch_size: int,
        drop_last_batch: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        optimizer: Optional[Union[Optimizer, List[Optimizer]]] = None,
        scheduler: Optional[Union[Callable, List[Callable]]] = None,
        criterion: Optional[Callable] = None,
        datasets: Optional[Dict[str, Optional[Dataset]]] = None,
        samplers: Optional[Dict[str, Optional[Sampler]]] = None,
        collate_fns: Optional[Dict[str, Optional[Callable]]] = None,
        metrics: Optional[Dict[str, Optional[Union[Metric, List[Metric]]]]] = None,
        postprocessing: Optional[Dict[str, Optional[Callable]]] = None,
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
        self.optimizer = ensure_list(optimizer)
        self.scheduler = ensure_list(scheduler)

        # DataLoader specifics
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Datasets, samplers, and collate functions
        schema = {"train": None, "val": None, "test": None, "predict": None}
        self.datasets = ensure_dict_schema(datasets, schema)
        self.samplers = ensure_dict_schema(samplers, schema)
        self.collate_fns = ensure_dict_schema(collate_fns, schema)

        # Metrics
        self.metrics = ensure_dict_schema(metrics, schema={"train": None, "val": None, "test": None})
        self.metrics = {mode: MetricCollection(ensure_list(metric)) for mode, metric in self.metrics.items()}
        # Register the metrics to allow the LightningModule to automatically move them to the correct device.
        # Currently, a workaround is needed because of https://github.com/pytorch/pytorch/issues/71203.
        # Once it's fixed, we can set `self.metrics = ModuleDict(self.metrics)` directly.
        for mode, mode_metrics in self.metrics.items():
            setattr(self, f"{mode}_metric", mode_metrics)
            self.metrics[mode] = getattr(self, f"{mode}_metric")

        # Postprocessing
        schema = {"input": None, "target": None, "pred": None}
        schema = {"criterion": schema, "metrics": schema, "logging": schema}
        self.postprocessing = ensure_dict_schema(postprocessing, schema)

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
        # Ensure that the batch is a list, a tuple, or a dict.
        if not isinstance(batch, (list, tuple, dict)):
            raise TypeError(
                "A batch must be a list, a tuple, or a dict."
                "A batch dict must have 'input' and 'target' as keys."
                "A batch list or a tuple must have 2 elements - input and target."
                "If target does not exist, return `None` as target."
            )
        # Ensure that a dict batch has input and target keys exclusively.
        if isinstance(batch, dict) and set(batch.keys()) != {"input", "target"}:
            raise ValueError("A batch must be a dict with 'input' and 'target' as keys.")
        # Ensure that a list/tuple batch has 2 elements (input and target).
        if len(batch) == 1:
            raise ValueError(
                "A batch must consist of 2 elements - input and target. If target does not exist, return `None` as target."
            )
        if len(batch) > 2:
            raise ValueError(f"A batch must consist of 2 elements - input and target, but found {len(batch)} elements.")

        # Split the batch into input and target.
        input, target = batch if not isinstance(batch, dict) else (batch["input"], batch["target"])

        # Forward
        if self.inferer and mode in ["val", "test", "predict"]:
            pred = self.inferer(input, self)
        else:
            pred = self(input)

        # Data postprocessing for criterion.
        input = apply_fns(input, self.postprocessing["criterion"]["input"])
        target = apply_fns(target, self.postprocessing["criterion"]["target"])
        pred = apply_fns(pred, self.postprocessing["criterion"]["pred"])

        # Calculate the loss.
        loss = self._calculate_loss(pred, target) if mode in ["train", "val"] else None
        # Log the loss for monitoring purposes.
        self.log(
            "loss" if mode == "train" else f"{mode}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            logger=False,
            batch_size=self.batch_size,
        )

        # Log and return the results.
        if mode == "predict":
            return pred
        else:
            # Data postprocessing for metrics
            input = apply_fns(input, self.postprocessing["metrics"]["input"])
            target = apply_fns(target, self.postprocessing["metrics"]["target"])
            pred = apply_fns(pred, self.postprocessing["metrics"]["pred"])

            # Calculate the metrics for the step.
            metrics = self.metrics[mode](pred, target)
            # Log the metrics for monitoring purposes.
            self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True, logger=False, batch_size=self.batch_size)

            # Data postprocessing for logging.
            input = apply_fns(input, self.postprocessing["logging"]["input"])
            target = apply_fns(target, self.postprocessing["logging"]["target"])
            pred = apply_fns(pred, self.postprocessing["logging"]["pred"])

            # Return the loss, metrics, input, target, and pred.
            return {"loss": loss, "metrics": metrics, "input": input, "target": target, "pred": pred}

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

    def configure_optimizers(self) -> Union[Optimizer, List[Dict[str, Union[Optimizer, "Scheduler"]]]]:
        """LightningModule method. Returns optimizers and, if defined, schedulers.

        Returns:
            Optimizer or a List of Dict of paired Optimizers and Schedulers: instantiated
                optimizers and/or schedulers.
        """
        if not self.optimizer:
            logger.error("Please specify 'system.optimizer' in the config. Exiting.")
            sys.exit()
        if not self.scheduler:
            return self.optimizer

        if len(self.optimizer) != len(self.scheduler):
            logger.error("Each optimizer must have its own scheduler.")
            sys.exit()

        return [{"optimizer": opt, "lr_scheduler": sched} for opt, sched in zip(self.optimizer, self.scheduler)]

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
        """`LighterSystem` dynamically defines the `..._dataloader()`and `..._step()` methods
        in the `self.setup()` method. However, when `LightningModule` excepts them to be defined
        at init. To prevent it from throwing an error, the `..._dataloader()` and `..._step()`
        are initially defined as `lambda: None`, before `self.setup()` is called.
        """
        self.train_dataloader = self.training_step = lambda: None
        self.val_dataloader = self.validation_step = lambda: None
        self.test_dataloader = self.test_step = lambda: None
        self.predict_dataloader = self.predict_step = lambda: None
