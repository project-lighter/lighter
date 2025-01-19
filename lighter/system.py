"""
This module defines the System class, which encapsulates the components of a deep learning system,
including the model, optimizer, datasets, and more. It extends PyTorch Lightning's LightningModule.
"""

from typing import Any, Callable

from dataclasses import asdict

import pytorch_lightning as pl
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import collate_str_fn, default_collate_fn_map
from torchmetrics import Metric, MetricCollection

from lighter.utils.misc import get_optimizer_stats, hasarg
from lighter.utils.types.containers import Adapters, DataLoaders, Metrics
from lighter.utils.types.enums import Data, Mode

# Patch the original collate function to allow None values in the batch. Done within the function to avoid global changes.
default_collate_fn_map.update({type(None): collate_str_fn})


class System(pl.LightningModule):
    """
    System encapsulates the components of a deep learning system, extending PyTorch Lightning's LightningModule.

    Args:
        model: Model.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        criterion: Criterion (loss) function.
        metrics: Metrics for train, val, and test. Supports a single or a list/dict of `torchmetrics` metrics.
        dataloaders: Dataloaders for train, val, test, and predict.
        adapters: TODO
        inferer: Inferer to use in val/test/predict modes.
            See MONAI inferers for more details: (https://docs.monai.io/en/stable/inferers.html).

    """

    def __init__(
        self,
        model: Module,
        dataloaders: dict[str, Dataset],
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        criterion: Callable | None = None,
        metrics: dict[str, Metric | list[Metric] | dict[str, Metric]] | None = None,
        adapters: dict[str, Callable] | None = None,
        inferer: Callable | None = None,
    ) -> None:
        super().__init__()

        # Model setup
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = Metrics(**(metrics or {}))
        self.adapters = Adapters(**(adapters or {}))
        self.inferer = inferer

        # Register metrics to move them to the appropriate device. ModuleDict not used because 'train' is a reserved key.
        for mode, metric in asdict(self.metrics).items():
            if isinstance(metric, Module):
                self.add_module(f"{Data.METRICS}_{mode}", metric)

        # Dataloader and step LightningModule methods
        dataloaders = DataLoaders(**(dataloaders or {}))
        if dataloaders.train is not None:
            self.train_dataloader = lambda: dataloaders.train
            self.training_step = self._step
        if dataloaders.val is not None:
            self.val_dataloader = lambda: dataloaders.val
            self.validation_step = self._step
        if dataloaders.test is not None:
            self.test_dataloader = lambda: dataloaders.test
            self.test_step = self._step
        if dataloaders.predict is not None:
            self.predict_dataloader = lambda: dataloaders.predict
            self.predict_step = self._step

        # Keep track of the current mode and its batch size. Overriden in on_{train,validation,test,predict}_start.
        self.mode = None
        self.batch_size = 0

    def _step(self, batch: dict, batch_idx: int) -> dict[str, Any] | Any:
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

        self._log_stats(loss=loss, metrics=metrics, batch_idx=batch_idx)
        output = self._prepare_output(input, target, pred, loss, metrics, identifier)
        return output

    def _prepare_batch(self, batch: dict) -> tuple[Any, Any, Any]:
        adapters = getattr(self.adapters, self.mode)
        input, target, identifier = adapters.batch(batch)
        return input, target, identifier

    def forward(self, input: Any) -> Any:  # pylint: disable=arguments-differ
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
        if hasarg(self.model.forward, Data.EPOCH):
            kwargs[Data.EPOCH] = self.current_epoch
        # Add `step` argument if forward accepts it
        if hasarg(self.model.forward, Data.STEP):
            kwargs[Data.STEP] = self.global_step

        # Use the inferer if specified and in val/test/predict mode
        if self.inferer and self.mode in [Mode.VAL, Mode.TEST, Mode.PREDICT]:
            return self.inferer(input, self.model, **kwargs)
        else:
            return self.model(input, **kwargs)

    def _calculate_loss(self, input: Any, target: Any, pred: Any) -> Tensor | dict[str, Tensor] | None:
        adapters = getattr(self.adapters, self.mode)
        loss = None
        if self.mode in [Mode.TRAIN, Mode.VAL]:
            loss = adapters.criterion(self.criterion, input=input, target=target, pred=pred)
            if isinstance(loss, dict) and "total" not in loss:
                raise ValueError(
                    "The loss dictionary must include a 'total' key that combines all sublosses. "
                    "Example: {'total': combined_loss, 'subloss1': loss1, ...}"
                )
        return loss

    def _calculate_metrics(self, input: Any, target: Any, pred: Any) -> Any | None:
        adapters = getattr(self.adapters, self.mode)
        metrics = getattr(self.metrics, self.mode)
        if metrics is not None:
            metrics = adapters.metrics(metrics, input=input, target=target, pred=pred)
        return metrics

    def _log_stats(self, loss: Tensor | dict[str, Tensor], metrics: MetricCollection, batch_idx: int) -> None:
        """
        Logs the loss, metrics, and optimizer statistics.

        Args:
            loss: The calculated loss.
            metrics: The calculated metrics.
            batch_idx: The index of the batch.
        """
        if self.trainer.logger is None:
            return

        # Loss
        if loss is not None:
            if not isinstance(loss, dict):
                self._log(f"{self.mode}/{Data.LOSS}/{Data.STEP}", loss, on_step=True)
                self._log(f"{self.mode}/{Data.LOSS}/{Data.EPOCH}", loss, on_epoch=True)
            else:
                for name, subloss in loss.items():
                    self._log(f"{self.mode}/{Data.LOSS}/{name}/{Data.STEP}", subloss, on_step=True)
                    self._log(f"{self.mode}/{Data.LOSS}/{name}/{Data.EPOCH}", subloss, on_epoch=True)
        # Metrics
        if metrics is not None:
            for name, metric in metrics.items():
                self._log(f"{self.mode}/{Data.METRICS}/{name}/{Data.STEP}", metric, on_step=True)
                self._log(f"{self.mode}/{Data.METRICS}/{name}/{Data.EPOCH}", metric, on_epoch=True)

        # Optimizer's lr, momentum, beta. Logged in train mode and once per epoch.
        if self.mode == Mode.TRAIN and batch_idx == 0:
            for name, optimizer_stat in get_optimizer_stats(self.optimizer).items():
                self._log(f"{self.mode}/{name}", optimizer_stat, on_epoch=True)

    def _log(self, name: str, value: Any, on_step: bool = False, on_epoch: bool = False) -> None:
        """Log a key, value pair. Syncs across distributed nodes if `on_epoch` is True.

        Args:
            name (str): key to log.
            value (Any): value to log.
            on_step (bool, optional): if True, logs on step.
            on_epoch (bool, optional): if True, logs on epoch with sync_dist=True.
        """
        self.log(name, value, logger=True, batch_size=self.batch_size, on_step=on_step, on_epoch=on_epoch, sync_dist=on_epoch)

    def _prepare_output(
        self,
        input: Any,
        target: Any,
        pred: Any,
        loss: Tensor | dict[str, Tensor] | None,
        metrics: Any | None,
        identifier: Any,
    ) -> dict[str, Any]:
        adapters = getattr(self.adapters, self.mode)
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
    def learning_rate(self, value: float) -> None:
        """
        Sets the learning rate of the optimizer.

        Args:
            value: The new learning rate.
        """
        if len(self.optimizer.param_groups) > 1:
            raise ValueError("The learning rate is not available when there are multiple optimizer parameter groups.")
        self.optimizer.param_groups[0]["lr"] = value

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

    def on_train_start(self) -> None:
        """Called when the train begins."""
        self.mode = Mode.TRAIN
        self.batch_size = self.train_dataloader().batch_size

    def on_validation_start(self) -> None:
        """Called when the validation loop begins."""
        self.mode = Mode.VAL
        self.batch_size = self.val_dataloader().batch_size

    def on_test_start(self) -> None:
        """Called when the test begins."""
        self.mode = Mode.TEST
        self.batch_size = self.test_dataloader().batch_size

    def on_predict_start(self) -> None:
        """Called when the prediction begins."""
        self.mode = Mode.PREDICT
        self.batch_size = self.predict_dataloader().batch_size
