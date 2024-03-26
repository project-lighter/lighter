from typing import Any, List, Optional, Union

from loguru import logger
from pytorch_lightning import Callback, Trainer
from torch.nn import Module

from lighter import LighterSystem
from lighter.utils.misc import ensure_list


class LighterFreezer(Callback):
    """
    Callback to freeze the parameters/layers of a model. Can be run indefinitely or until a specified step or epoch.
    `names` and`name_starts_with` can be used to specify which parameters to freeze.
    If both are specified, the parameters that match any of the two will be frozen.

    Args:
        names (str, List[str], optional): Names of the parameters to be frozen. Defaults to None.
        name_starts_with (str, List[str], optional): Prefixes of the parameter names to be frozen. Defaults to None.
        except_names (str, List[str], optional): Names of the parameters to be excluded from freezing. Defaults to None.
        except_name_starts_with (str, List[str], optional): Prefixes of the parameter names to be excluded from freezing.
            Defaults to None.
        until_step (int, optional): Maximum step to freeze parameters until. Defaults to None.
        until_epoch (int, optional): Maximum epoch to freeze parameters until. Defaults to None.

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
        Freezes the parameters of the model at the start of each training batch.

        Args:
            trainer (Trainer): Trainer instance.
            pl_module (LighterSystem): LighterSystem instance.
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
        Sets the requires_grad attribute of the model's parameters.

        Args:
            model (Module): PyTorch model whose parameters need to be frozen.
            requires_grad (bool): Whether to freeze the parameters or not.

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
