"""
This module defines the System class, which encapsulates the components of a deep learning system,
including the model, optimizer, datasets, and more. It extends PyTorch Lightning's LightningModule.
"""

from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate_str_fn, default_collate_fn_map
from torchmetrics import Metric

from lighter.flow import Flow
from lighter.utils.misc import get_optimizer_stats
from lighter.utils.patches import PatchedModuleDict
from lighter.utils.types.containers import DataLoaders, Flows, Metrics
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
        criterion: Criterion/loss function.
        metrics: Metrics for train, val, and test. Supports a single/list/dict of `torchmetrics` metrics.
        dataloaders: Dataloaders for train, val, test, and predict.
        flows: Flow objects that define the logic for train, val, test, and predict. See `lighter.flow.Flow`.

    """

    def __init__(
        self,
        model: Module,
        dataloaders: dict[str, DataLoader],
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        criterion: Callable | None = None,
        metrics: dict[str, Metric | list[Metric] | dict[str, Metric]] | None = None,
        flows: dict[str, Flow] | None = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        #  Containers
        self.dataloaders = DataLoaders(**(dataloaders or {}))
        self.metrics = Metrics(**(metrics or {}))
        self.flows = Flows(**(flows or {}))

        # Register metrics as a ModuleDict for proper device handling
        self.metrics = PatchedModuleDict(self.metrics.__dict__)

        # train/val/test/predict
        self.mode = None

        # Set up LightningModule hooks for train/val/test/predict
        self._setup_hooks()

    def _step(self, batch: dict, batch_idx: int) -> dict[str, Any] | Any:
        """
        Performs a step in the specified mode. It uses the `Flow` to process the batch,
        and returns the data dictionary, which includes loss, metrics, and predictions.

        Args:
            batch: The batch of data.
            batch_idx: The index of the batch.
        Returns:
            The data dictionary from the `Flow`. For training, it includes the loss.
            For prediction, it includes the predictions.
        """
        data = {Data.STEP: self.global_step, Data.EPOCH: self.current_epoch}

        metrics = self.metrics[self.mode]

        criterion = self.criterion if self.mode in [Mode.TRAIN, Mode.VAL] else None

        flow = getattr(self.flows, self.mode)

        data = flow(batch=batch, model=self.model, criterion=criterion, metrics=metrics, data=data)

        self._log(data, batch_idx)

        return data

    def _log(self, data: dict[str, Any], batch_idx: int) -> None:
        """
        Logs the loss, metrics, and optimizer statistics.

        Args:
            data: The data dictionary from the `_step` method.
            batch_idx: The index of the batch.
        """
        if self.trainer.logger is None:
            return

        dataloader = getattr(self.dataloaders, self.mode)
        batch_size = getattr(dataloader, "batch_size", None)

        def log(name: str, value: Any, on_step: bool = False, on_epoch: bool = False) -> None:
            """Log a key, value pair. Syncs across distributed nodes if `on_epoch` is True.

            Args:
                name (str): key to log.
                value (Any): value to log.
                on_step (bool, optional): if True, logs on step.
                on_epoch (bool, optional): if True, logs on epoch with sync_dist=True.
            """
            self.log(name, value, logger=True, batch_size=batch_size, on_step=on_step, on_epoch=on_epoch, sync_dist=on_epoch)

        # Loss
        loss = data.get(Data.LOSS)
        if loss is not None:
            if not isinstance(loss, dict):
                log(f"{self.mode}/{Data.LOSS}/{Data.STEP}", loss, on_step=True)
                log(f"{self.mode}/{Data.LOSS}/{Data.EPOCH}", loss, on_epoch=True)
            else:
                for name, subloss in loss.items():
                    log(f"{self.mode}/{Data.LOSS}/{name}/{Data.STEP}", subloss, on_step=True)
                    log(f"{self.mode}/{Data.LOSS}/{name}/{Data.EPOCH}", subloss, on_epoch=True)

        # Metrics
        metrics = data.get(Data.METRICS)
        if metrics is not None:
            for name, metric in metrics.items():
                log(f"{self.mode}/{Data.METRICS}/{name}/{Data.STEP}", metric, on_step=True)
                log(f"{self.mode}/{Data.METRICS}/{name}/{Data.EPOCH}", metric, on_epoch=True)

        # Optimizer's lr, momentum, beta. Logged in train mode and once per epoch.
        if self.mode == Mode.TRAIN and batch_idx == 0:
            for name, optimizer_stat in get_optimizer_stats(self.optimizer).items():
                log(f"{self.mode}/{name}", optimizer_stat, on_epoch=True)

    def configure_optimizers(self) -> dict[str, Optimizer | LRScheduler] | None:
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

    def _setup_hooks(self):
        """
        Sets up the LightningModule hooks for train/val/test/predict if the corresponding dataloaders are provided.
        """
        if self.dataloaders.train is not None:
            self.training_step = self._step
            self.train_dataloader = lambda: self.dataloaders.train
            self.on_train_start = lambda: self._on_mode_start(Mode.TRAIN)
        if self.dataloaders.val is not None:
            self.validation_step = self._step
            self.val_dataloader = lambda: self.dataloaders.val
            self.on_validation_start = lambda: self._on_mode_start(Mode.VAL)
        if self.dataloaders.test is not None:
            self.test_step = self._step
            self.test_dataloader = lambda: self.dataloaders.test
            self.on_test_start = lambda: self._on_mode_start(Mode.TEST)
        if self.dataloaders.predict is not None:
            self.predict_step = self._step
            self.predict_dataloader = lambda: self.dataloaders.predict
            self.on_predict_start = lambda: self._on_mode_start(Mode.PREDICT)

    def _on_mode_start(self, mode: str | None) -> None:
        """
        Sets the current mode at the start of a phase.

        Args:
            mode: The mode to set (train, val, test, or predict).
        """
        self.mode = mode

    @property
    def learning_rate(self) -> float:
        """
        Gets the learning rate of the optimizer.

        Returns:
            float: The learning rate.

        Raises:
            ValueError: If there are multiple optimizer parameter groups.
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

        Raises:
            ValueError: If there are multiple optimizer parameter groups.
        """
        if len(self.optimizer.param_groups) > 1:
            raise ValueError("The learning rate is not available when there are multiple optimizer parameter groups.")
        self.optimizer.param_groups[0]["lr"] = value
