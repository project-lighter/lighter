"""
This module provides the Freezer callback, which allows freezing model parameters during training.
"""

from typing import Any

from loguru import logger
from pytorch_lightning import Callback, LightningModule, Trainer

from lighter.utils.misc import ensure_list


class Freezer(Callback):
    """
    Callback to freeze model parameters during training. Parameters can be frozen by exact name or prefix.
    Freezing can be applied indefinitely or until a specified step/epoch.

    Args:
        names: Full names of parameters to freeze.
        name_starts_with: Prefixes of parameter names to freeze.
        except_names: Names of parameters to exclude from freezing.
        except_name_starts_with: Prefixes of parameter names to exclude from freezing.
        until_step: Maximum step to freeze parameters until.
        until_epoch: Maximum epoch to freeze parameters until.

    Raises:
        ValueError: If neither `names` nor `name_starts_with` are specified.
        ValueError: If both `until_step` and `until_epoch` are specified.

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

        self.names = ensure_list(names)
        self.name_starts_with = ensure_list(name_starts_with)
        self.except_names = ensure_list(except_names)
        self.except_name_starts_with = ensure_list(except_name_starts_with)
        self.until_step = until_step
        self.until_epoch = until_epoch

        self._frozen_state = False

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        """
        Called at the start of each training batch to freeze or unfreeze model parameters.

        Args:
            trainer: The trainer instance.
            pl_module: The LightningModule instance.
            batch: The current batch.
            batch_idx: The index of the batch.
        """
        current_step = trainer.global_step
        current_epoch = trainer.current_epoch

        # Unfreeze if the step or epoch limit has been reached.
        unfreeze_step = self.until_step is not None and current_step >= self.until_step
        unfreeze_epoch = self.until_epoch is not None and current_epoch >= self.until_epoch
        if unfreeze_step or unfreeze_epoch:
            if self._frozen_state:
                logger.info("Unfreezing the model.")
                self._set_model_requires_grad(pl_module, requires_grad=True)
                self._frozen_state = False
            return

        # Freeze if not already frozen.
        if not self._frozen_state:
            logger.info("Freezing the model.")
            self._set_model_requires_grad(pl_module, requires_grad=False)
            self._frozen_state = True

    def _set_model_requires_grad(self, model: LightningModule, requires_grad: bool) -> None:
        """
        Sets the `requires_grad` attribute for model parameters.

        When freezing (requires_grad=False):
        - Freeze specified parameters
        - Keep all others trainable (requires_grad=True)
        - Respect exception rules

        When unfreezing (requires_grad=True):
        - Unfreeze specified parameters
        - Keep all others trainable

        Args:
            model: The model whose parameters to modify.
            requires_grad: Whether to allow gradients (unfreeze) or not (freeze).
        """
        # If the model is a `LighterModule`, get the underlying network so users
        # can specify layer names without the "network." prefix.
        from lighter import LighterModule

        target = model.network if isinstance(model, LighterModule) else model

        frozen_layers = []
        unfrozen_layers = []

        for name, param in target.named_parameters():
            # Check if the parameter should be excluded from freezing.
            is_excepted = (self.except_names and name in self.except_names) or (
                self.except_name_starts_with and any(name.startswith(prefix) for prefix in self.except_name_starts_with)
            )
            if is_excepted:
                # Exceptions are always trainable
                param.requires_grad = True
                if not requires_grad:  # Only log when we're in freezing mode
                    unfrozen_layers.append(name)
                continue

            # Check if the parameter should be frozen/unfrozen.
            is_to_freeze = (self.names and name in self.names) or (
                self.name_starts_with and any(name.startswith(prefix) for prefix in self.name_starts_with)
            )
            if is_to_freeze:
                param.requires_grad = requires_grad
                if not requires_grad:
                    frozen_layers.append(name)
                else:
                    unfrozen_layers.append(name)
            else:
                # Not specified and not excepted - keep trainable
                param.requires_grad = True

        # Log the frozen/unfrozen layers.
        if frozen_layers:
            logger.info(
                f"Froze layers: {frozen_layers}"
                + (f" until step {self.until_step}" if self.until_step is not None else "")
                + (f" until epoch {self.until_epoch}" if self.until_epoch is not None else "")
            )
        if unfrozen_layers:
            suffix = " (excepted from freeze)" if not requires_grad else ""
            logger.info(f"Unfroze layers: {unfrozen_layers}{suffix}")
