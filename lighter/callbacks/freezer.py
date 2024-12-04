"""
This module provides the LighterFreezer callback, which allows freezing model parameters during training.
"""

from typing import Any, List, Optional, Union

from loguru import logger
from pytorch_lightning import Callback, Trainer
from torch.nn import Module

from lighter import LighterSystem
from lighter.utils.misc import ensure_list


class LighterFreezer(Callback):
    """
    Callback to freeze model parameters during training. Parameters can be frozen by exact name or prefix.
    Freezing can be applied indefinitely or until a specified step/epoch.

    Args:
        names (Optional[Union[str, List[str]]]): Full names of parameters to freeze.
        name_starts_with (Optional[Union[str, List[str]]]): Prefixes of parameter names to freeze.
        except_names (Optional[Union[str, List[str]]]): Names of parameters to exclude from freezing.
        except_name_starts_with (Optional[Union[str, List[str]]]): Prefixes of parameter names to exclude from freezing.
        until_step (int): Maximum step to freeze parameters until.
        until_epoch (int): Maximum epoch to freeze parameters until.

    Raises:
        ValueError: If neither `names` nor `name_starts_with` are specified.
        ValueError: If both `until_step` and `until_epoch` are specified.

    """

    def __init__(
        self,
        names: Optional[Union[str, List[str]]] = None,
        name_starts_with: Optional[Union[str, List[str]]] = None,
        except_names: Optional[Union[str, List[str]]] = None,
        except_name_starts_with: Optional[Union[str, List[str]]] = None,
        until_step: int = None,
        until_epoch: int = None,
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

    def on_train_batch_start(self, trainer: Trainer, pl_module: LighterSystem, batch: Any, batch_idx: int) -> None:
        """
        Called at the start of each training batch to potentially freeze parameters.

        Args:
            trainer (Trainer): The trainer instance.
            pl_module (LighterSystem): The LighterSystem instance.
            batch (Any): The current batch.
            batch_idx (int): The index of the batch.
        """
        self._on_batch_start(trainer, pl_module)

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module: LighterSystem, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._on_batch_start(trainer, pl_module)

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: LighterSystem, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._on_batch_start(trainer, pl_module)

    def on_predict_batch_start(
        self, trainer: Trainer, pl_module: LighterSystem, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._on_batch_start(trainer, pl_module)

    def _on_batch_start(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        """
        Freezes or unfreezes model parameters based on the current step or epoch.

        Args:
            trainer (Trainer): The trainer instance.
            pl_module (LighterSystem): The LighterSystem instance.
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

    def _set_model_requires_grad(self, model: Union[Module, LighterSystem], requires_grad: bool) -> None:
        """
        Sets the requires_grad attribute for model parameters, effectively freezing or unfreezing them.

        Args:
            model (Union[Module, LighterSystem]): The model whose parameters to modify.
            requires_grad (bool): Whether to allow gradients (unfreeze) or not (freeze).
        """
        # If the model is a `LighterSystem`, get the underlying PyTorch model.
        if isinstance(model, LighterSystem):
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
