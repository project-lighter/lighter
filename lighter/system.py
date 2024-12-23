"""
This module defines the LighterSystem class, which encapsulates the components of a deep learning system,
including the model, optimizer, datasets, and more. It extends PyTorch Lightning's LightningModule.
"""

from typing import Any, Callable, List, Tuple

from functools import partial

import pytorch_lightning as pl
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset
from torchmetrics import Metric, MetricCollection

from lighter.utils.enums import Mode
from lighter.utils.misc import apply_fns, get_optimizer_stats, hasarg
from lighter.utils.patches import PatchedModuleDict


class LighterSystem(pl.LightningModule):
    """
    LighterSystem encapsulates the components of a deep learning system, extending PyTorch Lightning's LightningModule.

    Args:
        model: Model.
        optimizer: Optimizers.
        scheduler: Learning rate scheduler.
        criterion: Criterion/loss function.
        inferer: Inferer must be a class with a `__call__` method that accepts two
            arguments - the input to infer over, and the model itself. Used in 'val', 'test', and 'predict'
            mode, but not in 'train'. Typically, an inferer is a sliding window or a patch-based inferer
            that will infer over the smaller parts of the input, combine them, and return a single output.
            The inferers provided by MONAI cover most of such cases (https://docs.monai.io/en/stable/inferers.html).
        metrics: Metrics for train, val, and test. Supports a single or a list/dict of `torchmetrics` metrics.
        dataloaders: Dataloaders for train, val, test, and predict.
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

    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        criterion: Callable | None = None,
        inferer: Callable | None = None,
        metrics: dict[str, Metric | List[Metric] | dict[str, Metric]] | None = None,
        dataloaders: dict[str, Dataset] | None = None,
        postprocessing: dict[str, Callable | List[Callable]] | None = None,
    ) -> None:
        super().__init__()

        # Model setup
        self.model = model

        # Criterion, optimizer, and scheduler
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Inferer for val, test, and predict
        self.inferer = inferer

        # Mode-specific metrics
        for mode, metric in metrics.items():
            metrics[mode] = MetricCollection(metric) if isinstance(metric, (list, dict)) else metric
        self.metrics = PatchedModuleDict(metrics)

        # Mode-specific dataloader and step methods
        if Mode.TRAIN in dataloaders:
            self.train_dataloader = lambda: dataloaders[Mode.TRAIN]
            self.train_step = partial(self._step, mode=Mode.TRAIN)
        if Mode.VAL in dataloaders:
            self.val_dataloader = lambda: dataloaders[Mode.VAL]
            self.validation_step = partial(self._step, mode=Mode.VAL)
        if Mode.TEST in dataloaders:
            self.test_dataloader = lambda: dataloaders[Mode.TEST]
            self.test_step = partial(self._step, mode=Mode.TEST)
        if Mode.PREDICT in dataloaders:
            self.predict_dataloader = lambda: dataloaders[Mode.PREDICT]
            self.predict_step = partial(self._step, mode=Mode.PREDICT)

        # Postprocessing
        self.postprocessing = postprocessing

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

    def _step(self, batch: dict, batch_idx: int, mode: str) -> dict[str, Any] | Any:
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
        if self.inferer and mode in [Mode.VAL, Mode.TEST, Mode.PREDICT]:
            pred = self.inferer(input, self)
        else:
            pred = self(input)

        # Postprocessing for loss calculation.
        # input = apply_fns(input, self.postprocessing["criterion"]["input"])
        # target = apply_fns(target, self.postprocessing["criterion"]["target"])
        # pred = apply_fns(pred, self.postprocessing["criterion"]["pred"])

        # Predict mode stops here.
        if mode == Mode.PREDICT:
            # Postprocessing for logging/writing.
            # pred = apply_fns(pred, self.postprocessing["logging"]["pred"])
            return {"pred": pred, "id": id}

        # Calculate the loss.
        loss = None
        if mode in [Mode.TRAIN, Mode.VAL]:
            # When target is not provided, pass only the predictions to the criterion.
            loss = self.criterion(pred) if target is None else self.criterion(pred, target)

        # Postprocessing for metrics.
        # input = apply_fns(input, self.postprocessing["metrics"]["input"])
        # target = apply_fns(target, self.postprocessing["metrics"]["target"])
        # pred = apply_fns(pred, self.postprocessing["metrics"]["pred"])

        # Calculate the step metrics.
        metrics = None
        if self.metrics[mode] is not None:
            metrics = self.metrics[mode](pred, target)

        # Postprocessing for logging/writing.
        # input = apply_fns(input, self.postprocessing["logging"]["input"])
        # target = apply_fns(target, self.postprocessing["logging"]["target"])
        # pred = apply_fns(pred, self.postprocessing["logging"]["pred"])

        # Ensure that a dict of losses has a 'total' key.
        if isinstance(loss, dict) and "total" not in loss:
            raise ValueError(
                "The loss dictionary must include a 'total' key that combines all sublosses. "
                "Example: {'total': combined_loss, 'subloss1': loss1, ...}"
            )

        # Logging
        self._log_stats(loss, metrics, mode, batch_idx)

        # Return the loss as required by Lightning and the other data to use in hooks or callbacks.
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
        if mode == Mode.TRAIN and batch_idx == 0:
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
