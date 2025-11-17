"""
This module defines the System class, which encapsulates the components of a deep learning system,
including the model, optimizer, datasets, and more. It extends PyTorch Lightning's LightningModule.
"""

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import pytorch_lightning as pl
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate_str_fn, default_collate_fn_map
from torchmetrics import Metric, MetricCollection

from lighter.utils.misc import get_optimizer_stats, hasarg
from lighter.utils.patches import PatchedModuleDict
from lighter.utils.types.containers import Adapters, DataLoaders, Metrics
from lighter.utils.types.enums import Data, Mode

# Patch the original collate function to allow None values in the batch.
default_collate_fn_map.update({type(None): collate_str_fn})


class System(pl.LightningModule):
    """
    System encapsulates the components of a deep learning system, extending PyTorch Lightning's LightningModule.

    Args:
        model: Model.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        criterion: Criterion (loss) function.
        metrics: Metrics for train, val, and test. Supports a single/list/dict of `torchmetrics` metrics.
        dataloaders: Dataloaders for train, val, test, and predict.
        adapters: Adapters for batch preparation, criterion argument adaptation, metrics argument adaptation, and logging data adaptation.
        inferer: Inferer to use in val/test/predict modes. Custom inferers can be defined to handle inference logic.

    """

    def __init__(
        self,
        model: Module,
        dataloaders: dict[str, DataLoader[Any]],
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        criterion: Callable[..., Any] | None = None,
        metrics: dict[str, Metric | list[Metric] | dict[str, Metric]] | None = None,
        adapters: dict[str, Callable[..., Any]] | None = None,
        inferer: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.inferer = inferer

        #  Containers
        self.dataloaders = DataLoaders(**(dataloaders or {}))
        self.metrics = Metrics(**(metrics or {}))  # type: ignore[arg-type]
        self.adapters = Adapters(**(adapters or {}))  # type: ignore[arg-type]

        # Turn metrics container into a ModuleDict to register them properly.
        self.metrics = PatchedModuleDict(asdict(self.metrics))  # type: ignore[assignment]

        self._setup_mode_hooks()

    def _step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any] | Any:
        """
        Performs a step in the specified mode, processing the batch and calculating loss and metrics.

        Args:
            batch: The batch of data.
            batch_idx: The index of the batch.
        Returns:
            dict or Any: For predict step, returns prediction only. For other steps,
            returns dict with loss, metrics, input, target, pred, and identifier. Loss is None
            for test step, metrics is None if unspecified.
        """
        input, target, identifier = self._prepare_batch(batch)
        pred = self.forward(input)

        loss = self._calculate_loss(input, target, pred)
        metrics = self._calculate_metrics(input, target, pred)

        self._log_stats(loss, metrics, batch_idx)
        output = self._prepare_output(identifier, input, target, pred, loss, metrics)
        return output

    def _get_current_mode(self) -> str:
        """
        Get the current execution mode from the trainer's state.

        Returns:
            The current mode (train, val, test, or predict).

        Raises:
            RuntimeError: If called outside of a trainer context or mode cannot be determined.
        """
        if self.trainer is None:
            raise RuntimeError("System must be attached to a Trainer to determine mode.")

        # During sanity checking, treat it as validation mode
        if self.trainer.sanity_checking:
            return Mode.VAL

        if self.trainer.training:
            return Mode.TRAIN
        elif self.trainer.validating:
            return Mode.VAL
        elif self.trainer.testing:
            return Mode.TEST
        elif self.trainer.predicting:
            return Mode.PREDICT
        else:
            raise RuntimeError(
                "Unable to determine current mode. This method should only be called "
                "during training, validation, testing, or prediction steps."
            )

    def _prepare_batch(self, batch: dict[str, Any]) -> tuple[Any, Any, Any]:
        """
        Prepares the batch data.

        Args:
            batch: The input batch dictionary.

        Returns:
            tuple: A tuple containing (input, target, identifier).
        """
        mode = self._get_current_mode()
        adapters = getattr(self.adapters, mode)  # type: ignore[arg-type]
        input, target, identifier = adapters.batch(batch)
        return input, target, identifier

    def forward(self, input: Any) -> Any:
        """
        Forward pass through the model.

        Args:
            input: The input data.

        Returns:
            Any: The model's output.
        """

        # Pass `epoch` and/or `step` argument to forward if it accepts them
        kwargs: dict[str, Any] = {}
        if hasarg(self.model.forward, Data.EPOCH):
            kwargs[Data.EPOCH] = self.current_epoch
        if hasarg(self.model.forward, Data.STEP):
            kwargs[Data.STEP] = self.global_step

        # Predict. Use inferer if available in val, test, and predict modes.
        mode = self._get_current_mode()
        if self.inferer and mode in [Mode.VAL, Mode.TEST, Mode.PREDICT]:
            return self.inferer(input, self.model, **kwargs)
        return self.model(input, **kwargs)

    def _calculate_loss(self, input: Any, target: Any, pred: Any) -> Tensor | dict[str, Tensor] | None:
        """
        Calculates the loss using the criterion if in train or validation mode.

        Args:
            input: The input data.
            target: The target data.
            pred: The model predictions.

        Returns:
            The calculated loss or None if not in train/val mode.

        Raises:
            ValueError: If criterion is not specified in train/val mode or if loss dict is missing 'total' key.
        """
        mode = self._get_current_mode()
        loss = None
        if mode in [Mode.TRAIN, Mode.VAL]:
            if self.criterion is None:
                raise ValueError("Please specify 'system.criterion' in the config.")

            adapters = getattr(self.adapters, mode)
            loss = adapters.criterion(self.criterion, input, target, pred)

            if isinstance(loss, dict) and "total" not in loss:
                raise ValueError(
                    "The loss dictionary must include a 'total' key that combines all sublosses. "
                    "Example: {'total': combined_loss, 'subloss1': loss1, ...}"
                )
        return loss

    def _calculate_metrics(self, input: Any, target: Any, pred: Any) -> Any | None:
        """
        Calculates the metrics if not in predict mode.

        Args:
            input: The input data.
            target: The target data.
            pred: The model predictions.

        Returns:
            The calculated metrics or None if in predict mode or no metrics specified.
        """
        mode = self._get_current_mode()
        if mode == Mode.PREDICT or self.metrics[mode] is None:  # type: ignore[index]
            return None

        adapters = getattr(self.adapters, mode)  # type: ignore[arg-type]
        metrics = adapters.metrics(self.metrics[mode], input, target, pred)  # type: ignore[index]
        return metrics

    def _log_stats(self, loss: Tensor | dict[str, Tensor] | None, metrics: MetricCollection | None, batch_idx: int) -> None:
        """
        Logs the loss, metrics, and optimizer statistics.

        Args:
            loss: The calculated loss.
            metrics: The calculated metrics.
            batch_idx: The index of the batch.
        """
        if self.trainer.logger is None:
            return

        mode = self._get_current_mode()

        # Loss
        if loss is not None:
            if not isinstance(loss, dict):
                self._log(f"{mode}/{Data.LOSS}/{Data.STEP}", loss, on_step=True)
                self._log(f"{mode}/{Data.LOSS}/{Data.EPOCH}", loss, on_epoch=True)
            else:
                for name, subloss in loss.items():
                    self._log(f"{mode}/{Data.LOSS}/{name}/{Data.STEP}", subloss, on_step=True)
                    self._log(f"{mode}/{Data.LOSS}/{name}/{Data.EPOCH}", subloss, on_epoch=True)

        # Metrics
        if metrics is not None:
            for name, metric in metrics.items():
                self._log(f"{mode}/{Data.METRICS}/{name}/{Data.STEP}", metric, on_step=True)
                self._log(f"{mode}/{Data.METRICS}/{name}/{Data.EPOCH}", metric, on_epoch=True)

        # Optimizer's lr, momentum, beta. Logged in train mode and once per epoch.
        if mode == Mode.TRAIN and batch_idx == 0 and self.optimizer is not None:
            for name, optimizer_stat in get_optimizer_stats(self.optimizer).items():
                self._log(f"{mode}/{name}", optimizer_stat, on_epoch=True)

    def _log(self, name: str, value: Any, on_step: bool = False, on_epoch: bool = False) -> None:
        """Log a key, value pair. Syncs across distributed nodes if `on_epoch` is True.

        Args:
            name (str): key to log.
            value (Any): value to log.
            on_step (bool, optional): if True, logs on step.
            on_epoch (bool, optional): if True, logs on epoch with sync_dist=True.
        """
        mode = self._get_current_mode()
        batch_size = getattr(self.dataloaders, mode).batch_size  # type: ignore[arg-type]
        self.log(name, value, logger=True, batch_size=batch_size, on_step=on_step, on_epoch=on_epoch, sync_dist=on_epoch)

    def _prepare_output(
        self,
        identifier: Any,
        input: Any,
        target: Any,
        pred: Any,
        loss: Tensor | dict[str, Tensor] | None,
        metrics: Any | None,
    ) -> dict[str, Any]:
        """
        Prepares the data to be returned by the step function to callbacks.

        Args:
            identifier: The batch identifier.
            input: The input data.
            target: The target data.
            pred: The model predictions.
            loss: The calculated loss.
            metrics: The calculated metrics.

        Returns:
            dict: A dictionary containing all the step information.
        """
        mode = self._get_current_mode()
        adapters = getattr(self.adapters, mode)  # type: ignore[arg-type]
        input, target, pred = adapters.logging(input, target, pred)
        return {
            Data.IDENTIFIER: identifier,
            Data.INPUT: input,
            Data.TARGET: target,
            Data.PRED: pred,
            Data.LOSS: loss,
            Data.METRICS: metrics,
            Data.STEP: self.global_step,
            Data.EPOCH: self.current_epoch,
        }

    def configure_optimizers(self) -> dict[str, Optimizer | LRScheduler] | None:  # type: ignore[override]
        """
        Configures the optimizers and learning rate schedulers.

        Returns:
            dict: A dictionary containing the optimizer and scheduler.

        Raises:
            ValueError: If optimizer is not specified.
        """
        if self.optimizer is None:
            raise ValueError("Please specify 'system.optimizer' in the config.")
        if self.scheduler is None:
            return {"optimizer": self.optimizer}
        else:
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def _setup_mode_hooks(self):
        """
        Sets up the training, validation, testing, and prediction hooks based on defined dataloaders.
        """
        if self.dataloaders.train is not None:
            self.training_step = self._step  # type: ignore[method-assign]
            self.train_dataloader = lambda: self.dataloaders.train  # type: ignore[method-assign]
        if self.dataloaders.val is not None:
            self.validation_step = self._step  # type: ignore[method-assign]
            self.val_dataloader = lambda: self.dataloaders.val  # type: ignore[method-assign]
        if self.dataloaders.test is not None:
            self.test_step = self._step  # type: ignore[method-assign]
            self.test_dataloader = lambda: self.dataloaders.test  # type: ignore[method-assign]
        if self.dataloaders.predict is not None:
            self.predict_step = self._step  # type: ignore[method-assign]
            self.predict_dataloader = lambda: self.dataloaders.predict  # type: ignore[method-assign]

    @property
    def learning_rate(self) -> float:
        """
        Gets the learning rate of the optimizer.

        Returns:
            float: The learning rate.

        Raises:
            ValueError: If there are multiple optimizer parameter groups.
            RuntimeError: If no optimizer is configured.
        """
        if self.optimizer is None:
            raise RuntimeError("No optimizer configured.")
        if len(self.optimizer.param_groups) > 1:
            raise ValueError("The learning rate is not available when there are multiple optimizer parameter groups.")
        lr: float = self.optimizer.param_groups[0]["lr"]
        return lr

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """
        Sets the learning rate of the optimizer.

        Args:
            value: The new learning rate.

        Raises:
            ValueError: If there are multiple optimizer parameter groups.
            RuntimeError: If no optimizer is configured.
        """
        if self.optimizer is None:
            raise RuntimeError("No optimizer configured.")
        if len(self.optimizer.param_groups) > 1:
            raise ValueError("The learning rate is not available when there are multiple optimizer parameter groups.")
        self.optimizer.param_groups[0]["lr"] = value
