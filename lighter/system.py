"""
This module defines the LighterSystem class, which encapsulates the components of a deep learning system,
including the model, optimizer, datasets, and more. It extends PyTorch Lightning's LightningModule.
"""

from typing import Any, Callable, List, Tuple

from functools import partial

import pytorch_lightning as pl
from loguru import logger
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Sampler
from torchmetrics import Metric, MetricCollection

from lighter.engine.schema import CollateFnSchema, DatasetSchema, MetricsSchema, PostprocessingSchema, SamplerSchema
from lighter.utils.collate import collate_replace_corrupted
from lighter.utils.misc import apply_fns, get_optimizer_stats, hasarg
from lighter.utils.patches import PatchedModuleDict


class LighterSystem(pl.LightningModule):
    """
    LighterSystem encapsulates the components of a deep learning system, extending PyTorch Lightning's LightningModule.

    Args:
        model: Model.
        batch_size: Batch size.
        drop_last_batch: Whether the last batch in the dataloader should be dropped.
        num_workers: Number of dataloader workers.
        pin_memory: Whether to pin the dataloaders memory.
        optimizer: Optimizers.
        scheduler: Learning rate scheduler.
        criterion: Criterion/loss function.
        datasets: Datasets for train, val, test, and predict.
        samplers: Samplers for train, val, test, and predict.
        collate_fns: Collate functions for train, val, test, and predict.
        metrics: Metrics for train, val, and test. Supports a single metric or a list/dict of `torchmetrics` metrics.

        postprocessing:
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
            and metrics earlier.
        inferer: Inferer must be a class with a `__call__` method that accepts two
            arguments - the input to infer over, and the model itself. Used in 'val', 'test', and 'predict'
            mode, but not in 'train'. Typically, an inferer is a sliding window or a patch-based inferer
            that will infer over the smaller parts of the input, combine them, and return a single output.
            The inferers provided by MONAI cover most of such cases (https://docs.monai.io/en/stable/inferers.html).

    """

    def __init__(
        self,
        model: Module,
        batch_size: int,
        drop_last_batch: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        criterion: Callable | None = None,
        datasets: dict[str, Dataset] | None = None,
        samplers: dict[str, Sampler] | None = None,
        collate_fns: dict[str, Callable | List[Callable]] | None = None,
        metrics: dict[str, Metric | List[Metric] | dict[str, Metric]] | None = None,
        postprocessing: dict[str, Callable | List[Callable]] | None = None,
        inferer: Callable | None = None,
    ) -> None:
        super().__init__()

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

        self.datasets = DatasetSchema(**(datasets or {})).model_dump()
        self.samplers = SamplerSchema(**(samplers or {})).model_dump()
        self.collate_fns = CollateFnSchema(**(collate_fns or {})).model_dump()
        self.postprocessing = PostprocessingSchema(**(postprocessing or {})).model_dump()
        self.metrics = MetricsSchema(**(metrics or {})).model_dump()

        self.metrics = PatchedModuleDict(self.metrics)

        # Inferer for val, test, and predict
        self.inferer = inferer

        # Bypasses LightningModule's check for dataloader and step methods. We define them dynamically in self.setup().
        self.train_dataloader = self.val_dataloader = self.test_dataloader = self.predict_dataloader = lambda: []
        self.training_step = self.validation_step = self.test_step = self.predict_step = lambda: None

    def forward(self, input: Tensor | List[Tensor] | Tuple[Tensor] | dict[str, Tensor]) -> Any:
        """
        Forward pass through the model. Supports multi-input models.

        Args:
            input: The input data.

        Returns:
            Any: The model's output.
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

    def configure_optimizers(self) -> dict:
        """
        Configures the optimizers and learning rate schedulers.

        Returns:
            dict: A dictionary containing the optimizer and scheduler.
        """
        if self.optimizer is None:
            raise ValueError("Please specify 'system.optimizer' in the config.")
        if self.scheduler is None:
            return {"optimizer": self.optimizer}
        else:
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def setup(self, stage: str) -> None:
        """
        Sets up the dataloaders and step methods for the specified stage.

        Args:
            stage: The stage of training (fit, validate, test, predict).
        """
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

    def _base_dataloader(self, mode: str) -> DataLoader:
        """
        Creates a DataLoader for the specified mode. Replaces None values (corrupted examples) in batches
        with valid examples using a custom collate function. Dataset should return None for corrupted data.

        Args:
            mode: The mode of operation (train, val, test, predict).

        Returns:
            DataLoader: The configured DataLoader.
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

    def _base_step(self, batch: dict, batch_idx: int, mode: str) -> dict[str, Any] | Any:
        """
        Performs a step in the specified mode, processing the batch and calculating loss and metrics.

        Args:
            batch: The batch of data.
            batch_idx: The index of the batch.
            mode: The mode of operation (train, val, test, predict).
        Returns:
            dict or Any: For predict step, returns prediction only. For other steps,
            returns dict with loss, metrics, input, target, pred, and id. Loss is None
            for test step, metrics is None if unspecified.
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

        # Calculate the step metrics.
        metrics = None
        if self.metrics[mode] is not None:
            metrics = self.metrics[mode](pred, target)

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

    def _log_stats(self, loss: Tensor | dict[str, Tensor], metrics: MetricCollection, mode: str, batch_idx: int) -> None:
        """
        Logs the loss, metrics, and optimizer statistics.

        Args:
            loss: The calculated loss.
            metrics: The calculated metrics.
            mode: The mode of operation (train, val, test, predict).
            batch_idx: The index of the batch.
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

    @property
    def learning_rate(self) -> float:
        """
        Gets the learning rate of the optimizer.

        Returns:
            float: The learning rate.
        """
        if len(self.optimizer.param_groups) > 1:
            raise ValueError("The learning rate is not available when there are multiple optimizer parameter groups.")
        return self.optimizer.param_groups[0]["lr"]

    @learning_rate.setter
    def learning_rate(self, value) -> None:
        """
        Sets the learning rate of the optimizer.

        Args:
            value: The new learning rate.
        """
        if len(self.optimizer.param_groups) > 1:
            raise ValueError("The learning rate is not available when there are multiple optimizer parameter groups.")
        self.optimizer.param_groups[0]["lr"] = value
