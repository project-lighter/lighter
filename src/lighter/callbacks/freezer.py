"""Freezer: freeze parameters by name during training."""

import logging
from typing import Any

from pytorch_lightning import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


def _as_list(value: str | list[str] | None) -> list[str]:
    return [value] if isinstance(value, str) else list(value or [])


class Freezer(Callback):
    """
    Freezes parameters by full name or name prefix, for the whole run or until a step or epoch.
    With a LighterModule, names are relative to its `network` (e.g. `encoder.weight`).

    Only the selected parameters are touched: unfreezing gives them back the `requires_grad` they had before.

    Args:
        names: Full names of parameters to freeze.
        name_starts_with: Prefixes of parameter names to freeze.
        except_names: Names of parameters to exclude from freezing.
        except_name_starts_with: Prefixes of parameter names to exclude from freezing.
        until_step: Maximum step to freeze parameters until.
        until_epoch: Maximum epoch to freeze parameters until.

    Raises:
        ValueError: If neither `names` nor `name_starts_with` are specified, if both `until_step` and
            `until_epoch` are, or if no parameter of the model matches.
    """

    def __init__(
        self,
        names: str | list[str] | None = None,
        name_starts_with: str | list[str] | None = None,
        except_names: str | list[str] | None = None,
        except_name_starts_with: str | list[str] | None = None,
        until_step: int | None = None,
        until_epoch: int | None = None,
    ) -> None:
        super().__init__()
        if names is None and name_starts_with is None:
            raise ValueError("At least one of `names` or `name_starts_with` must be specified.")
        if until_step is not None and until_epoch is not None:
            raise ValueError("Only one of `until_step` or `until_epoch` can be specified.")
        self.names = _as_list(names)
        self.name_starts_with = _as_list(name_starts_with)
        self.except_names = _as_list(except_names)
        self.except_name_starts_with = _as_list(except_name_starts_with)
        self.until_step = until_step
        self.until_epoch = until_epoch
        self._frozen: dict[str, bool] = {}  # the `requires_grad` of each frozen parameter before freezing
        self._done = False

    def _selects(self, name: str) -> bool:
        selected = name in self.names or name.startswith(tuple(self.name_starts_with))
        return selected and not (name in self.except_names or name.startswith(tuple(self.except_name_starts_with)))

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        """Freezes the selected parameters on the first batch, and unfreezes them once the step or epoch is reached."""
        from lighter.model import LighterModule

        reached = (self.until_step is not None and trainer.global_step >= self.until_step) or (
            self.until_epoch is not None and trainer.current_epoch >= self.until_epoch
        )
        if self._done or (self._frozen and not reached):
            return
        parameters = dict((pl_module.network if isinstance(pl_module, LighterModule) else pl_module).named_parameters())
        if reached:
            for name, requires_grad in self._frozen.items():
                parameters[name].requires_grad = requires_grad
            if self._frozen:
                logger.info(f"Unfroze {list(self._frozen)}.")
            self._frozen, self._done = {}, True
            return
        self._frozen = {name: parameter.requires_grad for name, parameter in parameters.items() if self._selects(name)}
        if not self._frozen:
            raise ValueError(f"Freezer matched no parameter. The model's parameter names are like {list(parameters)[:5]}.")
        for name in self._frozen:
            parameters[name].requires_grad = False
        until = f" until step {self.until_step}" if self.until_step is not None else ""
        until = f" until epoch {self.until_epoch}" if self.until_epoch is not None else until
        logger.info(f"Froze {list(self._frozen)}{until}.")
