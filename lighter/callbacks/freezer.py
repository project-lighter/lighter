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
        names (str, List[str], optional): The names of the parameters to be frozen. Defaults to None.
        name_starts_with (str, List[str], optional): The prefixes of the parameter names to be frozen. Defaults to None.
        except_names (str, List[str], optional): The names of the parameters to be excluded from freezing. Defaults to None.
        except_name_starts_with (str, List[str], optional): The prefixes of the parameter names to be excluded from freezing.
            Defaults to None.
        until_step (int, optional): The maximum step to freeze parameters until. Defaults to None.
        until_epoch (int, optional): The maximum epoch to freeze parameters until. Defaults to None.

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

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LighterSystem, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self._on_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module: LighterSystem, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self._on_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: LighterSystem, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self._on_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_predict_batch_start(
        self, trainer: Trainer, pl_module: LighterSystem, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self._on_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def _on_batch_start(
        self, trainer: Trainer, pl_module: LighterSystem, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """
        Freezes the parameters of the model at the start of each training batch.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer instance.
            pl_module (LighterSystem): The PyTorch Lightning module instance.
            batch (Any): The current batch data.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int): The index of the current dataloader.

        """
        current_step = trainer.global_step
        current_epoch = trainer.current_epoch

        if self.until_step is not None and current_step > self.until_step:
            if self._frozen_state:
                self._set_model_requires_grad(pl_module, True)
            return

        if self.until_epoch is not None and current_epoch > self.until_epoch:
            if self._frozen_state:
                self._set_model_requires_grad(pl_module, True)
            return

        if not self._frozen_state:
            self._set_model_requires_grad(pl_module, False)

    def _set_model_requires_grad(self, model: Module, requires_grad: bool) -> None:
        """
        Sets the requires_grad attribute of the model's parameters.

        Args:
            model (Module): The PyTorch model whose parameters need to be frozen.
            requires_grad (bool): Whether to freeze the parameters or not.

        """
        frozen_layers = []
        # Freeze the specified parameters.
        for name, param in model.named_parameters():
            # Skip the parameters that are excluded from freezing.
            if self.except_names and name in self.except_names:
                param.requires_grad = True
            elif self.except_name_starts_with and any(name.startswith(prefix) for prefix in self.except_name_starts_with):
                param.requires_grad = True
            # Freeze the specified parameters.
            elif self.names and name in self.names:
                param.requires_grad = requires_grad
                frozen_layers.append(name)
            elif self.name_starts_with and any(name.startswith(prefix) for prefix in self.name_starts_with):
                param.requires_grad = requires_grad
                frozen_layers.append(name)
            else:
                param.requires_grad = True

        logger.info(
            f"Setting requires_grad={requires_grad} the following layers"
            + (f" until step {self.until_step}" if self.until_step is not None else "")
            + (f" until epoch {self.until_epoch}" if self.until_epoch is not None else "")
            + f": {frozen_layers}"
        )

        self._frozen_state = not requires_grad