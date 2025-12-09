"""
This module provides the core LighterModule class that extends PyTorch Lightning's LightningModule.
Users implement abstract step methods while the framework handles automatic dual logging.
"""

from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Metric, MetricCollection

from lighter.utils.misc import get_optimizer_stats
from lighter.utils.types.enums import Mode


class LighterModule(pl.LightningModule):
    """
    Minimal base class for deep learning models in Lighter.

    Users should:
    - Subclass and implement the step methods they need (training_step, validation_step, etc.)
    - Define their own batch processing, loss computation, metric updates
    - Configure data separately using the 'data:' config key

    Framework provides:
    - Automatic dual logging of losses (step + epoch)
    - Automatic dual logging of metrics (step + epoch)
    - Optimizer configuration

    Args:
        network: Neural network model
        criterion: Loss function (optional, user can compute loss manually in step)
        optimizer: Optimizer (required for training)
        scheduler: Learning rate scheduler (optional)
        train_metrics: Training metrics (optional, user calls them in step)
        val_metrics: Validation metrics (optional)
        test_metrics: Test metrics (optional)

    Example:
        class MyModel(LighterModule):
            def training_step(self, batch, batch_idx):
                x, y = batch
                pred = self(x)

                # Option 1: Use self.criterion if provided
                loss = self.criterion(pred, y) if self.criterion else F.cross_entropy(pred, y)

                # User calls metrics themselves
                if self.train_metrics:
                    self.train_metrics(pred, y)

                return {"loss": loss, "pred": pred, "target": y}

            def validation_step(self, batch, batch_idx):
                x, y = batch
                pred = self(x)
                loss = self.criterion(pred, y) if self.criterion else F.cross_entropy(pred, y)
                if self.val_metrics:
                    self.val_metrics(pred, y)
                return {"loss": loss, "pred": pred, "target": y}

            def test_step(self, batch, batch_idx):
                x, y = batch
                pred = self(x)
                if self.test_metrics:
                    self.test_metrics(pred, y)
                return {"pred": pred, "target": y}

            def predict_step(self, batch, batch_idx):
                x, y = batch
                pred = self(x)
                return pred
    """

    def __init__(
        self,
        network: Module,
        criterion: Callable | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        train_metrics: Metric | MetricCollection | None = None,
        val_metrics: Metric | MetricCollection | None = None,
        test_metrics: Metric | MetricCollection | None = None,
    ) -> None:
        super().__init__()

        # Core components
        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Metrics (registered as modules)
        self.train_metrics = self._prepare_metrics(train_metrics)
        self.val_metrics = self._prepare_metrics(val_metrics)
        self.test_metrics = self._prepare_metrics(test_metrics)

    def _prepare_metrics(self, metrics: Metric | MetricCollection | None) -> Metric | MetricCollection | None:
        """Validate metrics - must be Metric or MetricCollection."""
        if metrics is None:
            return None

        if isinstance(metrics, (Metric, MetricCollection)):
            return metrics

        raise TypeError(
            f"metrics must be Metric or MetricCollection, got {type(metrics).__name__}.\n\n"
            f"Single metric:\n"
            f"  train_metrics:\n"
            f"    _target_: torchmetrics.Accuracy\n"
            f"    task: multiclass\n\n"
            f"Multiple metrics:\n"
            f"  train_metrics:\n"
            f"    _target_: torchmetrics.MetricCollection\n"
            f"    metrics:\n"
            f"      - _target_: torchmetrics.Accuracy\n"
            f"        task: multiclass\n"
            f"      - _target_: torchmetrics.F1Score\n"
            f"        task: multiclass"
        )

    # ============================================================================
    # Step Methods - Override as Needed
    # ============================================================================

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor | dict[str, Any]:
        """
        Define training logic.

        User responsibilities:
        - Extract data from batch
        - Call self(input) for forward pass
        - Compute loss
        - Call self.train_metrics(pred, target) if configured
        - Return loss tensor or dict with 'loss' key

        Framework automatically logs loss and metrics.

        Returns:
            Either:
                - Tensor: The loss value (simplest option)
                - Dict with required 'loss' key and optional keys:
                    - pred: Model predictions (for callbacks)
                    - target: Target labels (for callbacks)
                    - input: Input data (for callbacks)
                    - Any other keys you need
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement training_step() to use trainer.fit(). "
            f"See https://project-lighter.github.io/lighter/guides/lighter-module/"
        )

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor | dict[str, Any]:
        """
        Define validation logic.

        Similar to training_step but typically without gradients.
        Call self.val_metrics(pred, target) if configured.

        Returns:
            Either:
                - Tensor: The loss value
                - Dict with 'loss' key
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement validation_step() to use validation. "
            f"See https://project-lighter.github.io/lighter/guides/lighter-module/"
        )

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor | dict[str, Any]:
        """
        Define test logic.

        Loss is optional. Call self.test_metrics(pred, target) if configured.

        Returns:
            Either:
                - Tensor: The loss value (optional in test mode)
                - Dict with optional 'loss' key. Can include pred, target, etc.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement test_step() to use trainer.test(). "
            f"See https://project-lighter.github.io/lighter/guides/lighter-module/"
        )

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        """
        Define prediction logic.

        User responsibilities:
        - Extract data from batch
        - Call self(input) for forward pass
        - Return predictions in desired format

        No automatic logging happens in predict mode.
        Return any format you need (tensor, dict, list, etc.).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement predict_step() to use trainer.predict(). "
            f"See https://project-lighter.github.io/lighter/guides/lighter-module/"
        )

    # ============================================================================
    # Forward Pass - Simple Delegation
    # ============================================================================

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward pass - simply delegates to self.network.

        Override if you need custom forward logic.
        """
        return self.network(*args, **kwargs)

    # ============================================================================
    # Batch-End Hooks - Automatic Logging
    # ============================================================================

    def _on_batch_end(self, outputs: torch.Tensor | dict[str, Any], batch_idx: int) -> None:
        """Common batch-end logic for all modes."""
        outputs = self._normalize_output(outputs)
        self._log_outputs(outputs, batch_idx)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Framework hook - automatically logs training outputs."""
        self._on_batch_end(outputs, batch_idx)

    def on_validation_batch_end(
        self,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Framework hook - automatically logs validation outputs."""
        self._on_batch_end(outputs, batch_idx)

    def on_test_batch_end(
        self,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Framework hook - automatically logs test outputs."""
        self._on_batch_end(outputs, batch_idx)

    def _normalize_output(self, output: torch.Tensor | dict[str, Any]) -> dict[str, Any]:
        """
        Normalize step output to dict format.

        Args:
            output: Either:
                - torch.Tensor: Loss value (normalized to {"loss": tensor})
                - dict: Must contain outputs. Can include:
                    - "loss": torch.Tensor or dict with "total" key
                    - "pred", "target", "input": Additional data for callbacks

        Returns:
            Dict with normalized structure

        Raises:
            TypeError: If output is neither Tensor nor dict
            ValueError: If loss dict is missing 'total' key
        """
        if isinstance(output, torch.Tensor):
            return {"loss": output}
        elif isinstance(output, dict):
            # Validate loss structure if present
            if "loss" in output and isinstance(output["loss"], dict):
                if "total" not in output["loss"]:
                    raise ValueError(
                        f"Loss dict must include 'total' key. "
                        f"Got keys: {list(output['loss'].keys())}. "
                        f"Example: {{'loss': {{'total': combined, 'ce': ce_loss, 'reg': reg_loss}}}}"
                    )
            return output
        else:
            raise TypeError(
                f"Step method must return torch.Tensor or dict. "
                f"Got {type(output).__name__} instead. "
                f"Examples:\n"
                f"  - return loss  # Simple tensor\n"
                f'  - return {{"loss": loss, "pred": pred}}'
            )

    def _log_outputs(self, outputs: dict[str, Any], batch_idx: int) -> None:
        """
        Log all outputs from a step.

        Override this method to customize logging behavior.
        Default: dual logging (step + epoch) for loss and metrics.

        Args:
            outputs: Dict from user's step method
            batch_idx: Current batch index
        """
        if self.trainer.logger is None:
            return
        self._log_loss(outputs.get("loss"))
        self._log_metrics()
        self._log_optimizer_stats(batch_idx)

    def _log_loss(self, loss: torch.Tensor | dict[str, Any] | None) -> None:
        """
        Log loss with dual pattern (step + epoch).

        Args:
            loss: Loss tensor or dict from step method.
                If dict, must have 'total' key (validated in _normalize_output).
        """
        if loss is None:
            return

        # Log scalar or dict
        if isinstance(loss, dict):
            for name, value in loss.items():
                name = f"{self.mode}/loss/{name}"
                self._log(name, value, on_step=True)
                self._log(name, value, on_epoch=True, sync_dist=True)
        else:
            name = f"{self.mode}/loss"
            self._log(name, loss, on_step=True)
            self._log(name, loss, on_epoch=True, sync_dist=True)

    def _log_metrics(self) -> None:
        """
        Log metrics with dual pattern (step + epoch).

        User already called metrics in their step method.
        Handles both single Metric and MetricCollection.
        """
        metrics = getattr(self, f"{self.mode}_metrics", None)
        if metrics is None:
            return

        if isinstance(metrics, MetricCollection):
            # MetricCollection - iterate over named metrics
            for name, metric in metrics.items():
                name = f"{self.mode}/metrics/{name}"
                self._log(name, metric, on_step=True)
                self._log(name, metric, on_epoch=True, sync_dist=True)
        else:
            # Single Metric - use class name (consistent with MetricCollection auto-naming)
            name = f"{self.mode}/metrics/{metrics.__class__.__name__}"
            self._log(name, metrics, on_step=True)
            self._log(name, metrics, on_epoch=True, sync_dist=True)

    def _log_optimizer_stats(self, batch_idx: int) -> None:
        """
        Log optimizer stats once per epoch in train mode.

        Args:
            batch_idx: Current batch index
        """
        if self.mode != Mode.TRAIN or batch_idx != 0 or self.optimizer is None:
            return

        # Optimizer stats only logged per epoch
        for name, stat in get_optimizer_stats(self.optimizer).items():
            name = f"{self.mode}/{name}"
            self._log(name, stat, on_epoch=True, sync_dist=False)

    def _log(self, name: str, value: Any, on_step: bool = False, on_epoch: bool = False, sync_dist: bool = False) -> None:
        suffix = "step" if on_step and not on_epoch else "epoch"
        self.log(
            f"{name}/{suffix}",
            value,
            logger=True,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=sync_dist,
        )

    # ============================================================================
    # Lightning Optimizer Configuration
    # ============================================================================

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        if self.optimizer is None:
            raise ValueError("Optimizer not configured.")

        if self.scheduler is None:
            return {"optimizer": self.optimizer}
        else:
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    # ============================================================================
    # Properties
    # ============================================================================

    @property
    def mode(self) -> str:
        """
        Current execution mode.

        Returns:
            "train", "val", "test", or "predict"

        Raises:
            RuntimeError: If called outside trainer context
        """
        if self.trainer is None:
            raise RuntimeError("LighterModule is not attached to a Trainer.")

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
            raise RuntimeError("Cannot determine mode outside Lightning execution.")
