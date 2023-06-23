from typing import List, Optional, Union

from loguru import logger
from torch.nn import Module

from lighter.utils.misc import ensure_list


class LighterFreezer:
    """
    Freezes the parameters/layers of a model. Can be run indefinitely or until a specified step or epoch.
    `names` and`name_starts_with` can be used to specify which parameters to freeze.
    If both are specified, the parameters that match any of the two will be frozen.

    Args:
        names (str, List[str], optional): The names of the parameters to be frozen. Defaults to None.
        name_starts_with (str, List[str], optional): The prefixes of the parameter names to be frozen. Defaults to None.
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
        until_step: int = None,
        until_epoch: int = None,
    ) -> None:
        if names is None and name_starts_with is None:
            raise ValueError("At least one of `names` or `name_starts_with` must be specified.")

        if until_step is not None and until_epoch is not None:
            raise ValueError("Only one of `until_step` or `until_epoch` can be specified.")

        self.names = ensure_list(names)
        self.name_starts_with = ensure_list(name_starts_with)
        self.until_step = until_step
        self.until_epoch = until_epoch

        self._frozen_state = False

    def __call__(self, model: Module, current_step: int, current_epoch: int) -> None:
        """
        Freezes the parameters of the model.

        Args:
            model (Module): The PyTorch model whose parameters need to be frozen.
            current_step (int): The current step.
            current_epoch (int): The current epoch.

        """
        if self.until_step is not None and current_step > self.until_step:
            if self._frozen_state:
                self.set_model_requires_grad(model, True)
            return

        if self.until_epoch is not None and current_epoch > self.until_epoch:
            if self._frozen_state:
                self.set_model_requires_grad(model, True)
            return

        if not self._frozen_state:
            self.set_model_requires_grad(model, False)

    def set_model_requires_grad(self, model, requires_grad):
        """
        Sets the requires_grad attribute of the model's parameters.

        Args:
            model (Module): The PyTorch model whose parameters need to be frozen.
            requires_grad (bool): Whether to freeze the parameters or not.

        """
        frozen_layers = []
        # Freeze the specified parameters.
        for name, param in model.named_parameters():
            if self.names and name in self.names:
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

        self._frozen_state = not (requires_grad)
