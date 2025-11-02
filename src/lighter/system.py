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
        criterion: Criterion (loss) function.
        metrics: Metrics for train, val, and test. Supports a single/list/dict of `torchmetrics` metrics.
        dataloaders: Dataloaders for train, val, test, and predict.
        flows: Flow objects that define the logic for each step (train, val, test, predict).

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
        self.metrics = PatchedModuleDict(Metrics(**(metrics or {})).__dict__)
        self.flows = Flows(**(flows or {}))

        self.mode = None
        self._setup_mode_hooks()

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
        flow = getattr(self.flows, self.mode)
        context = {Data.STEP: self.global_step, Data.EPOCH: self.current_epoch}
        output = flow(
            batch=batch, model=self.model, criterion=self.criterion, metrics=self.metrics.get(self.mode), context=context
        )

        self._log_stats(output, batch_idx)
        return output

    def _log_stats(self, output: dict[str, Any], batch_idx: int) -> None:
        """
        Logs the loss, metrics, and optimizer statistics.

        Args:
            output: The output dictionary from the `_step` method.
            batch_idx: The index of the batch.
        """
        if self.trainer.logger is None:
            return

        # Loss
        loss = output.get(Data.LOSS)
        if loss is not None:
            if not isinstance(loss, dict):
                self._log(f"{self.mode}/{Data.LOSS}/{Data.STEP}", loss, on_step=True)
                self._log(f"{self.mode}/{Data.LOSS}/{Data.EPOCH}", loss, on_epoch=True)
            else:
                for name, subloss in loss.items():
                    self._log(f"{self.mode}/{Data.LOSS}/{name}/{Data.STEP}", subloss, on_step=True)
                    self._log(f"{self.mode}/{Data.LOSS}/{name}/{Data.EPOCH}", subloss, on_epoch=True)

        # Metrics
        metrics = output.get(Data.METRICS)
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
        batch_size = getattr(self.dataloaders, self.mode).batch_size
        self.log(name, value, logger=True, batch_size=batch_size, on_step=on_step, on_epoch=on_epoch, sync_dist=on_epoch)

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

    def _setup_mode_hooks(self):
        """
        Sets up the training, validation, testing, and prediction hooks based on defined dataloaders.
        """
        if self.dataloaders.train is not None:
            self.training_step = self._step
            self.train_dataloader = lambda: self.dataloaders.train
            self.on_train_start = lambda: self._on_mode_start(Mode.TRAIN)
            self.on_train_end = self._on_mode_end
        if self.dataloaders.val is not None:
            self.validation_step = self._step
            self.val_dataloader = lambda: self.dataloaders.val
            self.on_validation_start = lambda: self._on_mode_start(Mode.VAL)
            self.on_validation_end = self._on_mode_end
        if self.dataloaders.test is not None:
            self.test_step = self._step
            self.test_dataloader = lambda: self.dataloaders.test
            self.on_test_start = lambda: self._on_mode_start(Mode.TEST)
            self.on_test_end = self._on_mode_end
        if self.dataloaders.predict is not None:
            self.predict_step = self._step
            self.predict_dataloader = lambda: self.dataloaders.predict
            self.on_predict_start = lambda: self._on_mode_start(Mode.PREDICT)
            self.on_predict_end = self._on_mode_end

    def _on_mode_start(self, mode: str | None) -> None:
        """
        Sets the current mode at the start of a phase.

        Args:
            mode: The mode to set (train, val, test, or predict).
        """
        self.mode = mode

    def _on_mode_end(self) -> None:
        """
        Resets the mode at the end of a phase.
        """
        self.mode = None

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
