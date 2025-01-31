"""
This module provides the Freezer callback, which allows freezing model parameters during training.
"""

from typing import Any, List

from loguru import logger
from pytorch_lightning import Callback, Trainer
from torch.nn import Module

from lighter import System
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
        names: str | List[str] | None = None,
        name_starts_with: str | List[str] | None = None,
        except_names: str | List[str] | None = None,
        except_name_starts_with: str | List[str] | None = None,
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

    def on_train_batch_start(self, trainer: Trainer, pl_module: System, batch: Any, batch_idx: int) -> None:
        """
        Called at the start of each training batch to potentially freeze parameters.

        Args:
            trainer: The trainer instance.
            pl_module: The System instance.
            batch: The current batch.
            batch_idx: The index of the batch.
        """
        self._on_batch_start(trainer, pl_module)

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module: System, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._on_batch_start(trainer, pl_module)

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: System, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._on_batch_start(trainer, pl_module)

    def on_predict_batch_start(
        self, trainer: Trainer, pl_module: System, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._on_batch_start(trainer, pl_module)

    def _on_batch_start(self, trainer: Trainer, pl_module: System) -> None:
        """
        Freezes or unfreezes model parameters based on the current step or epoch.

        Args:
            trainer: The trainer instance.
            pl_module: The System instance.
        """
        current_step = trainer.global_step
        current_epoch = trainer.current_epoch

        if self.until_step is not None and current_step >= self.until_step:
            if self._frozen_state:
                logger.info(f"Reached step {self.until_step} - unfreezing the previously frozen layers.")
                self._set_model_requires_grad(pl_module, True)
            return

        if self.until_epoch is not None and current_epoch >= self.until_epoch:
            if self._frozen_state:
                logger.info(f"Reached epoch {self.until_epoch} - unfreezing the previously frozen layers.")
                self._set_model_requires_grad(pl_module, True)
            return

        if not self._frozen_state:
            self._set_model_requires_grad(pl_module, False)

    def _set_model_requires_grad(self, model: Module | System, requires_grad: bool) -> None:
        """
        Sets the requires_grad attribute for model parameters, effectively freezing or unfreezing them.

        Args:
            model: The model whose parameters to modify.
            requires_grad: Whether to allow gradients (unfreeze) or not (freeze).
        """
        # If the model is a `System`, get the underlying PyTorch model.
        if isinstance(model, System):
            model = model.model

        frozen_layers = []
        # Freeze the specified parameters.
        for name, param in model.named_parameters():
            # Leave the excluded-from-freezing parameters trainable.
            if self.except_names and name in self.except_names:
                param.requires_grad = True
                continue
            if self.except_name_starts_with and any(name.startswith(prefix) for prefix in self.except_name_starts_with):
                param.requires_grad = True
                continue

            # Freeze/unfreeze the specified parameters, based on the `requires_grad` argument.
            if self.names and name in self.names:
                param.requires_grad = requires_grad
                frozen_layers.append(name)
                continue
            if self.name_starts_with and any(name.startswith(prefix) for prefix in self.name_starts_with):
                param.requires_grad = requires_grad
                frozen_layers.append(name)
                continue

            # Otherwise, leave the parameter trainable.
            param.requires_grad = True

        self._frozen_state = not requires_grad
        # Log only when freezing the parameters.
        if self._frozen_state:
            logger.info(
                f"Setting requires_grad={requires_grad} the following layers"
                + (f" until step {self.until_step}" if self.until_step is not None else "")
                + (f" until epoch {self.until_epoch}" if self.until_epoch is not None else "")
                + f": {frozen_layers}"
            )
