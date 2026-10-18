"""LighterModule: a LightningModule whose parts come from the config and whose losses and metrics are logged for you."""

from collections.abc import Callable, Mapping
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_warn
from torchmetrics import Metric, MetricCollection

LOSS_DICT = (
    "`loss` must be one tensor, the total that is backpropagated. Return {'loss': total} and log the parts "
    "yourself, e.g. self.log('train/loss/ce', ce)."
)


class LighterModule(pl.LightningModule):
    """A LightningModule built from the config: write the step methods you need, as in Lightning.

    A step returns the loss tensor, a dict with a `"loss"` key (other keys are passed on to callbacks), or None.
    Lighter logs the loss as `<mode>/loss` and the `<mode>_metrics` you updated in the step as
    `<mode>/metrics/<name>`, each per step (`/step`) and per epoch (`/epoch`), for train, val and test.

    Args:
        network: The model that `forward` calls.
        criterion: A loss function for your steps to use (optional).
        optimizer: An optimizer, or a function that builds one from the parameters
            (`_mode_: callable` in the config).
        scheduler: A scheduler, a function that builds one from the optimizer (`_mode_: callable`),
            or Lightning's scheduler dict (`scheduler`, `interval`, `monitor`, ...) holding either.
        train_metrics: A torchmetrics `Metric` or `MetricCollection` (optional).
        val_metrics: The same, for validation.
        test_metrics: The same, for testing.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        criterion: Callable | None = None,
        optimizer: Any = None,
        scheduler: Any = None,
        train_metrics: Metric | MetricCollection | None = None,
        val_metrics: Metric | MetricCollection | None = None,
        test_metrics: Metric | MetricCollection | None = None,
    ) -> None:
        super().__init__()
        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        for name, metrics in (("train_metrics", train_metrics), ("val_metrics", val_metrics), ("test_metrics", test_metrics)):
            if metrics is not None and not isinstance(metrics, (Metric, MetricCollection)):
                raise TypeError(
                    f"`{name}` must be a torchmetrics Metric or MetricCollection, got {type(metrics).__name__}. "
                    "For several metrics, use `_target_: torchmetrics.MetricCollection` with `metrics: [...]`."
                )
            setattr(self, name, metrics)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.network(*args, **kwargs)

    def configure_optimizers(self) -> Any:
        """The optimizer (built from `self.parameters()` if it is a function) and the optional scheduler."""
        if self.optimizer is None:
            raise ValueError("LighterModule needs an optimizer to train: set `model::optimizer` in the config.")
        optimizer = self.optimizer(self.parameters()) if callable(self.optimizer) else self.optimizer
        if self.scheduler is None:
            return optimizer
        scheduler = dict(self.scheduler) if isinstance(self.scheduler, Mapping) else {"scheduler": self.scheduler}
        if callable(scheduler["scheduler"]):
            scheduler["scheduler"] = scheduler["scheduler"](optimizer)
        elif optimizer is not self.optimizer:
            raise ValueError(
                "The scheduler was built before the optimizer it should drive. The optimizer has `_mode_: callable`, "
                "so give the scheduler `_mode_: callable` too."
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        self._log_step("train", outputs)

    def on_validation_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self._log_step("val", outputs, dataloader_idx)

    def on_test_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self._log_step("test", outputs, dataloader_idx)

    def _log_step(self, mode: str, outputs: Any, dataloader_idx: int = 0) -> None:
        """Log the loss a step returned and the `<mode>_metrics`, per step and per epoch."""
        loss = outputs.get("loss") if isinstance(outputs, Mapping) else outputs
        if isinstance(loss, Mapping):
            raise TypeError(LOSS_DICT)
        if isinstance(loss, bool) or not isinstance(loss, (torch.Tensor, int, float, type(None))):
            rank_zero_warn(
                f"The {mode} step returned {type(outputs).__name__}, so no loss is logged. Return the loss tensor, "
                "a dict with a 'loss' key, or None."
            )
            loss = None
        elif mode == "train" and self.automatic_optimization and loss is not None:
            loss = loss * self.trainer.accumulate_grad_batches  # Lightning passes the loss divided by it
        metrics = getattr(self, f"{mode}_metrics")
        if metrics is not None and dataloader_idx > 0:
            raise ValueError(
                f"`{mode}_metrics` would mix the results of several dataloaders. Use one dataloader, or create a "
                f"metric per dataloader in your module and log each with self.log."
            )
        named = metrics.items() if isinstance(metrics, MetricCollection) else [(type(metrics).__name__, metrics)]
        for name, value in [("loss", loss), *((f"metrics/{name}", metric) for name, metric in named)]:
            if value is not None:
                self.log(f"{mode}/{name}/step", value, on_step=True, on_epoch=False)
                self.log(f"{mode}/{name}/epoch", value, on_step=False, on_epoch=True, sync_dist=True)
