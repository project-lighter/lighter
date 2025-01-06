"""
This module defines the System class, which encapsulates the components of a deep learning system,
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
        metrics: dict[str, Metric | List[Metric] | dict[str, Metric]] | None = None,
        adapters: dict[str, Callable] | None = None,
        inferer: Callable | None = None,
    ) -> None:
        super().__init__()

        # Model setup
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.inferer = inferer

        # Metrics
        self.metrics = Metrics(**(metrics or {}))

        # Adapters
        self.adapters = Adapters(**(adapters or {}))

        # Set up methods for dataloaders and steps
        dataloaders = DataLoaders(**(dataloaders or {}))
        if dataloaders.train is not None:
            self.train_dataloader = lambda: dataloaders.train
            self.training_step = partial(self._step, mode=Mode.TRAIN)
        if dataloaders.val is not None:
            self.val_dataloader = lambda: dataloaders.val
            self.validation_step = partial(self._step, mode=Mode.VAL)
        if dataloaders.test is not None:
            self.test_dataloader = lambda: dataloaders.test
            self.test_step = partial(self._step, mode=Mode.TEST)
        if dataloaders.predict is not None:
            self.predict_dataloader = lambda: dataloaders.predict
            self.predict_step = partial(self._step, mode=Mode.PREDICT)

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
            returns dict with loss, metrics, input, target, pred, and identifier. Loss is None
            for test step, metrics is None if unspecified.
        """
        adapters = getattr(self.adapters, mode)

        # Batch adapter formats the batch into the required format
        input = adapters.batch.input(batch)
        target = adapters.batch.target(batch)
        identifier = adapters.batch.identifier(batch)

        # Forward
        if self.inferer and mode in [Mode.VAL, Mode.TEST, Mode.PREDICT]:
            pred = self.inferer(input, self)
        else:
            pred = self(input)

        # Predict mode stops here.
        if mode == Mode.PREDICT:
            # Logging adapter applies any specied transforms to for pred data
            return {Data.IDENTIFIER: identifier, Data.PRED: adapters.logging.pred(pred)}

        # Calculate the loss.
        loss = None
        if mode in [Mode.TRAIN, Mode.VAL]:
            loss = adapters.criterion(self.criterion, input=input, target=target, pred=pred)
            if isinstance(loss, dict) and "total" not in loss:
                raise ValueError(
                    "The loss dictionary must include a 'total' key that combines all sublosses. "
                    "Example: {'total': combined_loss, 'subloss1': loss1, ...}"
                )

        # Calculate the metrics. If no metrics are specified, the returned metrics value is None.
        metrics = getattr(self.metrics, mode)
        if metrics is not None:
            metrics = adapters.metrics(metrics, input=input, target=target, pred=pred)

        self._log_stats(loss=loss, metrics=metrics, mode=mode, batch_idx=batch_idx)

        # Return Lightning-required loss and additional data for callbacks like logging
        return {
            "loss": loss["total"] if isinstance(loss, dict) else loss,
            "metrics": metrics,
            Data.IDENTIFIER: identifier,
            Data.INPUT: adapters.logging.input(input),
            Data.TARGET: adapters.logging.target(target),
            Data.PRED: adapters.logging.pred(pred),
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

        # Loss
        if loss is not None:
            if not isinstance(loss, dict):
                self._log(f"{mode}/loss/step", loss, on_step=True)
                self._log(f"{mode}/loss/epoch", loss, on_epoch=True)
            else:
                for name, subloss in loss.items():
                    self._log(f"{mode}/loss/{name}/step", subloss, on_step=True)
                    self._log(f"{mode}/loss/{name}/epoch", subloss, on_epoch=True)
        # Metrics
        if metrics is not None:
            for name, metric in metrics.items():
                self._log(f"{mode}/metrics/{name}/step", metric, on_step=True)
                self._log(f"{mode}/metrics/{name}/epoch", metric, on_epoch=True)

        # Optimizer's lr, momentum, beta. Logged in train mode and once per epoch.
        if mode == Mode.TRAIN and batch_idx == 0:
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
        self.log(name, value, logger=True, batch_size=self.batch_size, on_step=on_step, on_epoch=on_epoch, sync_dist=on_epoch)

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
