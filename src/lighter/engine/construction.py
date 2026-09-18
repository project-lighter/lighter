"""Private lifecycle binding for compatible Runner-managed Lighter modules."""

import inspect
from collections.abc import Iterator
from dataclasses import dataclass, field
from pydoc import locate
from typing import Any

from sparkwheel import Component
from sparkwheel.construction import RetainedConfig
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from lighter.model import LighterModule

_FIELDS = {name for name in inspect.signature(LighterModule.__init__).parameters if name != "self"}
_DEFERRED = {"optimizer", "scheduler"}


def _has_owned_input(value: Any) -> bool:
    """Opaque optimizer inputs retain their explicitly supplied lifetime."""
    if isinstance(value, (Optimizer, LRScheduler, ReduceLROnPlateau, Iterator, Parameter)):
        return True
    if isinstance(value, dict):
        return any(_has_owned_input(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_owned_input(child) for child in value)
    return False


@dataclass
class _OptimizerBinding:
    recipe: RetainedConfig
    bindings: dict[str, Any]
    blocked_paths: set[str]
    fields: dict[str, Any]
    deferred: set[str]
    components: dict[str, Any] = field(default_factory=dict)
    locations: list[tuple[str, str, str, Any, bool]] = field(default_factory=list)

    def finalize(self, components: dict[str, Any]) -> None:
        """Retain intentional shared targets, never the run resolver/cache."""
        for path, value in components.items():
            if isinstance(value, (Optimizer, LRScheduler, ReduceLROnPlateau)):
                raise ValueError(
                    f"Managed optimizer setup conflicts with an optimizer/scheduler constructed early at '{path}'. "
                    "Keep optimizer-dependent construction in native hooks or use custom module ownership."
                )
        self.components = dict(components)
        self.locations.clear()
        for field_name, owner in self.fields.items():
            if not isinstance(owner, Module):
                continue
            modules = list(owner.named_modules(remove_duplicate=False))
            parameters = list(owner.named_parameters(remove_duplicate=False))
            for source_path, value in components.items():
                if isinstance(value, Module):
                    self.locations.extend(
                        (source_path, field_name, name, value, False) for name, child in modules if child is value
                    )
                elif isinstance(value, Parameter):
                    self.locations.extend(
                        (source_path, field_name, name, value, True) for name, child in parameters if child is value
                    )

    def build(self, model: LighterModule) -> None:
        for name, original in self.fields.items():
            if getattr(model, name) is not original:
                raise ValueError(
                    f"Managed construction cannot rebind replaced field 'model::{name}' without invalidating shared aliases. "
                    "Start a new run or use a native/custom configure_optimizers ownership path."
                )
        for source_path, field_name, name, original, parameter in self.locations:
            owner = getattr(model, field_name)
            try:
                current = owner.get_parameter(name) if parameter else owner.get_submodule(name)
            except AttributeError:
                current = None
            if current is not original:
                raise ValueError(
                    f"Managed component alias '{source_path}' no longer matches the live model::{field_name}.{name}. "
                    "Use native/custom optimizer ownership for replacing shared constructed components."
                )

        bindings = {**self.bindings, **self.components, "model": model}
        bindings.update({f"model::{name}": getattr(model, name) for name in self.fields})
        scope = self.recipe.scope(bindings=bindings, blocked_paths=self.blocked_paths)
        model.optimizer = scope.resolve("model::optimizer") if "optimizer" in self.deferred else None
        model.scheduler = scope.resolve("model::scheduler") if "scheduler" in self.deferred else None


def resolve_managed_model(view: Any) -> LighterModule | None:
    """Select a narrow managed profile without evaluating speculative targets."""
    definition = view.recipe.definition("model")
    if not isinstance(definition, dict) or "_target_" not in definition:
        return None
    if definition.get("_mode_", "default") != "default" or definition.get("_disabled_", False) is not False:
        return None
    if definition.get("_args_") or set(definition) - Component.non_arg_keys - _FIELDS:
        return None
    raw_target = definition["_target_"]
    if isinstance(raw_target, str):
        if raw_target.startswith(("$", "@", "%")):
            return None
        target = locate(raw_target)
    else:
        target = raw_target
    if not isinstance(target, type) or not issubclass(target, LighterModule):
        return None
    if target.__init__ is not LighterModule.__init__ or target.configure_optimizers is not LighterModule.configure_optimizers:
        return None
    # A conservative source-only boundary includes aliases to opaque inputs.
    # Do not inspect live object internals or evaluate inactive definitions.
    if _has_owned_input(view.recipe.definition()):
        return None

    deferred = set(definition) & _DEFERRED
    blocked = view.blocked_paths | {f"model::{name}" for name in deferred}
    view.scope = view.recipe.scope(bindings=view.bindings, blocked_paths=blocked)
    values = {
        name: view.scope.resolve(f"model::{name}")
        for name in definition
        if name not in Component.non_arg_keys and name not in _DEFERRED
    }
    model = target(**values)
    view.scope.bind("model", model)
    model._optimizer_binding = _OptimizerBinding(view.recipe, dict(view.bindings), set(view.blocked_paths), values, deferred)
    view.managed_binding = model._optimizer_binding
    return model
