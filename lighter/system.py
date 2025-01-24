"""
This module defines the System class, which encapsulates the components of a deep learning system,
including the model, optimizer, datasets, and more. It extends PyTorch Lightning's LightningModule.
"""

from typing import Any, Callable

from dataclasses import asdict
from functools import partial

import pytorch_lightning as pl
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data._utils.collate import collate_str_fn, default_collate_fn_map
from torchmetrics import Metric, MetricCollection

from lighter.utils.data import DataLoader
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
        metrics: Metrics for train, val, and test. Supports a single/list/dict of `torchmetrics` metrics.
        dataloaders: Dataloaders for train, val, test, and predict.
        adapters: TODO
        inferer: Inferer to use in val/test/predict modes.
            See MONAI inferers for more details: (https://docs.monai.io/en/stable/inferers.html).

    """

    def __init__(
        self,
        model: Module,
        dataloaders: dict[str, DataLoader],
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
        self.dataloaders = DataLoaders(**(dataloaders or {}))
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.metrics = Metrics(**(metrics or {}))
        self.adapters = Adapters(**(adapters or {}))
        self.inferer = inferer

        self._register_metrics()
        self._setup_steps_and_dataloaders()

    def _step(self, batch: dict, batch_idx: int, mode: str) -> dict[str, Any] | Any:
        """
        Performs a step in the specified mode, processing the batch and calculating loss and metrics.

        Args:
            batch: The batch of data.
            batch_idx: The index of the batch.
            mode: The mode of the step (train, val, test, predict).
        Returns:
            dict or Any: For predict step, returns prediction only. For other steps,
            returns dict with loss, metrics, input, target, pred, and identifier. Loss is None
            for test step, metrics is None if unspecified.
        """
        input, target, identifier = self._prepare_batch(batch, mode)
        pred = self.forward(input, mode)

        loss = self._calculate_loss(input, target, pred, mode)
        metrics = self._calculate_metrics(input, target, pred, mode)

        self._log_stats(loss, metrics, batch_idx, mode)
        output = self._prepare_output(identifier, input, target, pred, loss, metrics, mode)
        return output

    def _prepare_batch(self, batch: dict, mode: str) -> tuple[Any, Any, Any]:
        adapters = getattr(self.adapters, mode)
        input, target, identifier = adapters.batch(batch)
        return input, target, identifier

    def forward(self, input: Any, mode: str) -> Any:  # pylint: disable=arguments-differ
        """
        Forward pass through the model. Supports multi-input models.

        Args:
            input: The input data.
            mode: The mode of the forward pass (train, val, test, predict).

        Returns:
            Any: The model's output.
        """

        # Pass `epoch` and/or `step` argument to forward if it accepts them
        kwargs = {}
        if hasarg(self.model.forward, Data.EPOCH):
            kwargs[Data.EPOCH] = self.current_epoch
        if hasarg(self.model.forward, Data.STEP):
            kwargs[Data.STEP] = self.global_step

        # Predict. Use inferer if available in val, test, and predict modes.
        if self.inferer and mode in [Mode.VAL, Mode.TEST, Mode.PREDICT]:
            return self.inferer(input, self.model, **kwargs)
        return self.model(input, **kwargs)

    def _calculate_loss(self, input: Any, target: Any, pred: Any, mode: str) -> Tensor | dict[str, Tensor] | None:
        adapters = getattr(self.adapters, mode)
        loss = None
        if mode in [Mode.TRAIN, Mode.VAL]:
            loss = adapters.criterion(self.criterion, input, target, pred)
            if isinstance(loss, dict) and "total" not in loss:
                raise ValueError(
                    "The loss dictionary must include a 'total' key that combines all sublosses. "
                    "Example: {'total': combined_loss, 'subloss1': loss1, ...}"
                )
        return loss

    def _calculate_metrics(self, input: Any, target: Any, pred: Any, mode: str) -> Any | None:
        adapters = getattr(self.adapters, mode)
        metrics = getattr(self.metrics, mode)
        if metrics is not None:
            metrics = adapters.metrics(metrics, input, target, pred)
        return metrics

    def _log_stats(self, loss: Tensor | dict[str, Tensor], metrics: MetricCollection, batch_idx: int, mode: str) -> None:
        """
        Logs the loss, metrics, and optimizer statistics.

        Args:
            loss: The calculated loss.
            metrics: The calculated metrics.
            batch_idx: The index of the batch.
            mode: The mode of the step (train, val, test, predict).
        """
        if self.trainer.logger is None:
            return

        batch_size = getattr(self.dataloaders, mode).batch_size

        # Loss
        if loss is not None:
            if not isinstance(loss, dict):
                self._log(f"{mode}/{Data.LOSS}/{Data.STEP}", loss, batch_size, on_step=True)
                self._log(f"{mode}/{Data.LOSS}/{Data.EPOCH}", loss, batch_size, on_epoch=True)
            else:
                for name, subloss in loss.items():
                    self._log(f"{mode}/{Data.LOSS}/{name}/{Data.STEP}", subloss, batch_size, on_step=True)
                    self._log(f"{mode}/{Data.LOSS}/{name}/{Data.EPOCH}", subloss, batch_size, on_epoch=True)
        # Metrics
        if metrics is not None:
            for name, metric in metrics.items():
                self._log(f"{mode}/{Data.METRICS}/{name}/{Data.STEP}", metric, batch_size, on_step=True)
                self._log(f"{mode}/{Data.METRICS}/{name}/{Data.EPOCH}", metric, batch_size, on_epoch=True)

        # Optimizer's lr, momentum, beta. Logged in train mode and once per epoch.
        if mode == Mode.TRAIN and batch_idx == 0:
            for name, optimizer_stat in get_optimizer_stats(self.optimizer).items():
                self._log(f"{mode}/{name}", optimizer_stat, batch_size, on_epoch=True)

    def _log(self, name: str, value: Any, batch_size, on_step: bool = False, on_epoch: bool = False) -> None:
        """Log a key, value pair. Syncs across distributed nodes if `on_epoch` is True.

        Args:
            name (str): key to log.
            value (Any): value to log.
            batch_size (int): batch size.
            on_step (bool, optional): if True, logs on step.
            on_epoch (bool, optional): if True, logs on epoch with sync_dist=True.
        """
        self.log(name, value, logger=True, batch_size=batch_size, on_step=on_step, on_epoch=on_epoch, sync_dist=on_epoch)

    def _prepare_output(
        self,
        identifier: Any,
        input: Any,
        target: Any,
        pred: Any,
        loss: Tensor | dict[str, Tensor] | None,
        metrics: Any | None,
        mode: str,
    ) -> dict[str, Any]:
        adapters = getattr(self.adapters, mode)
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

    def _register_metrics(self):
        # Register metrics to move them to the appropriate device. ModuleDict not used because 'train' is a reserved key.
        for mode, metric in asdict(self.metrics).items():
            if isinstance(metric, Module):
                self.add_module(f"{Data.METRICS}_{mode}", metric)

    def _setup_steps_and_dataloaders(self):
        # Dataloader and step methods. Defined only if the corresponding dataloader is provided.
        if self.dataloaders.train is not None:
            self.train_dataloader = lambda: self.dataloaders.train(Mode.TRAIN)
            self.training_step = partial(self._step, mode=Mode.TRAIN)
        if self.dataloaders.val is not None:
            self.val_dataloader = lambda: self.dataloaders.val(Mode.VAL)
            self.validation_step = partial(self._step, mode=Mode.VAL)
        if self.dataloaders.test is not None:
            self.test_dataloader = lambda: self.dataloaders.test(Mode.TEST)
            self.test_step = partial(self._step, mode=Mode.TEST)
        if self.dataloaders.predict is not None:
            self.predict_dataloader = lambda: self.dataloaders.predict(Mode.PREDICT)
            self.predict_step = partial(self._step, mode=Mode.PREDICT)

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
