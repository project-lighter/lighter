"""Selected-parameter freezing using native Lightning callbacks and checkpoints."""

import json
from typing import Any

import torch
from loguru import logger
from pytorch_lightning import Callback, LightningModule, Trainer
from torch.nn.parallel import DistributedDataParallel

from lighter.utils.misc import ensure_list


class Freezer(Callback):
    """Freeze selected parameters, then restore their original gradient flags.

    Names are exact parameter names; prefixes use ``name_starts_with``. For a
    LighterModule, names are relative to its network. Exceptions exclude ownership:
    unrelated and excepted parameters are never made trainable by this callback.
    Limits are native optimizer steps or epochs; at the limit, original flags are
    restored. Parameters that may reactivate must already belong to an optimizer.
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
        for value in (until_step, until_epoch):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Freezer limits must be nonnegative integers")
        self.names = ensure_list(names)
        self.name_starts_with = ensure_list(name_starts_with)
        self.except_names = ensure_list(except_names)
        self.except_name_starts_with = ensure_list(except_name_starts_with)
        for selector in (self.names, self.name_starts_with, self.except_names, self.except_name_starts_with):
            if any(not isinstance(name, str) or not name for name in selector):
                raise ValueError("Freezer selectors must be nonempty strings")
        self.until_step = until_step
        self.until_epoch = until_epoch
        self._original_flags: dict[str, bool] = {}
        self._frozen_state: bool | None = None

    @property
    def state_key(self) -> str:
        """Distinguish independent callback policies in native checkpoints."""
        policy = {
            "names": sorted(self.names),
            "prefixes": sorted(self.name_starts_with),
            "except_names": sorted(self.except_names),
            "except_prefixes": sorted(self.except_name_starts_with),
            "until_step": self.until_step,
            "until_epoch": self.until_epoch,
        }
        return f"{type(self).__qualname__}:{json.dumps(policy, sort_keys=True)}"

    def state_dict(self) -> dict[str, Any]:
        return {"original_flags": dict(self._original_flags)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        flags = state_dict.get("original_flags", {})
        if not isinstance(flags, dict) or any(
            not isinstance(name, str) or type(flag) is not bool for name, flag in flags.items()
        ):
            raise ValueError("Invalid Freezer checkpoint original_flags")
        self._original_flags = dict(flags)
        # requires_grad is not restored by a native module state_dict. Reapply it.
        self._frozen_state = None

    def _selected_parameters(self, model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
        from lighter.model import LighterModule

        target = model.network if isinstance(model, LighterModule) else model
        canonical: dict[int, str] = {}
        selected: dict[str, torch.nn.Parameter] = {}
        excluded: set[int] = set()
        for name, parameter in target.named_parameters(remove_duplicate=False):
            canonical.setdefault(id(parameter), name)
            if name in self.except_names or any(name.startswith(prefix) for prefix in self.except_name_starts_with):
                excluded.add(id(parameter))
            if name in self.names or any(name.startswith(prefix) for prefix in self.name_starts_with):
                selected[canonical[id(parameter)]] = parameter
        selected = {name: parameter for name, parameter in selected.items() if id(parameter) not in excluded}
        if not selected:
            raise ValueError("Freezer selection matched no parameters; names are exact, use name_starts_with for prefixes")
        return selected

    def _capture_original_flags(self, selected: dict[str, torch.nn.Parameter]) -> None:
        if self._original_flags and self._original_flags.keys() != selected.keys():
            raise ValueError("Freezer parameter names changed since setup/checkpoint; use a matching model and policy")
        if not self._original_flags:
            self._original_flags = {name: parameter.requires_grad for name, parameter in selected.items()}

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        selected = self._selected_parameters(pl_module)
        self._capture_original_flags(selected)
        # Lightning 2.5 initializes this public attribute through its connector.
        callbacks: list[Callback] = trainer.callbacks  # type: ignore[attr-defined]
        for callback in callbacks:
            if isinstance(callback, Freezer) and callback is not self:
                other_ids = {id(parameter) for parameter in callback._selected_parameters(pl_module).values()}
                if any(id(parameter) in other_ids for parameter in selected.values()):
                    raise ValueError("Freezer callbacks overlap in parameter ownership; combine their policy")
        dynamic = self.until_step is not None or self.until_epoch is not None
        if dynamic:
            optimized = {
                id(parameter)
                for optimizer in trainer.optimizers
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            missing = [
                name for name, parameter in selected.items() if self._original_flags[name] and id(parameter) not in optimized
            ]
            if missing:
                raise ValueError(f"Freezer parameters that can reactivate must already belong to an optimizer: {missing}")
        wrapped = trainer.strategy.model
        if isinstance(wrapped, DistributedDataParallel):
            if not wrapped.find_unused_parameters or wrapped.static_graph:
                raise ValueError("Freezer with DDP requires find_unused_parameters=True and static_graph=False")
            if dynamic and any(
                self._original_flags[name] and not parameter.requires_grad for name, parameter in selected.items()
            ):
                raise ValueError("Reactivated Freezer parameters must be trainable when DDP wraps the model")
        elif trainer.world_size > 1:
            raise ValueError(
                "Freezer supports distributed execution through DDP; other strategies require a native freezing policy"
            )
        self._frozen_state = None

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        released = (self.until_step is not None and trainer.global_step >= self.until_step) or (
            self.until_epoch is not None and trainer.current_epoch >= self.until_epoch
        )
        frozen = not released
        if frozen != self._frozen_state:
            self._set_model_requires_grad(pl_module, requires_grad=released)
            self._frozen_state = frozen

    def _set_model_requires_grad(self, model: torch.nn.Module, requires_grad: bool) -> None:
        """Apply ownership: false freezes, true restores each original flag."""
        selected = self._selected_parameters(model)
        self._capture_original_flags(selected)
        for name, parameter in selected.items():
            parameter.requires_grad_(self._original_flags[name] if requires_grad else False)
            if not parameter.requires_grad:
                parameter.grad = None
        logger.info(f"{'Restored gradient flags for' if requires_grad else 'Froze'} parameters: {list(selected)}")
