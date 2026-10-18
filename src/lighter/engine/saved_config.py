"""Save each run's composed config, and rebuild a model from the config saved in its checkpoint."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback

from lighter.engine.runner import _find_up, load_config, to_yaml
from lighter.utils.dynamic_imports import import_module_from_path


class SaveConfig(Callback):
    """Added to every trainer Lighter builds: saves the composed config with the run.

    On the global-zero process it writes `config.yaml` to the log folder and logs the config as
    hyperparameters, once the log folder is final (also under DDP). Every checkpoint stores the config
    under `"lighter_config"`, which `load_model` reads.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:
        if trainer.is_global_zero and trainer.logger:
            trainer.logger.log_hyperparams(self.config)
            if trainer.log_dir:
                Path(trainer.log_dir).mkdir(parents=True, exist_ok=True)
                (Path(trainer.log_dir) / "config.yaml").write_text(to_yaml(self.config))

    def on_save_checkpoint(self, trainer: Any, pl_module: Any, checkpoint: dict) -> None:
        checkpoint["lighter_config"] = self.config


def load_model(ckpt_path: str | Path, overrides: Iterable[str] = (), weights_only: bool = True) -> Any:
    """Rebuild the model of a Lighter run from the config in its checkpoint, and load the checkpoint's weights.

    Args:
        ckpt_path: A checkpoint saved by a run that Lighter built.
        overrides: Overrides applied to the saved config first, e.g. `["model::network::dropout=0"]`.
        weights_only: Load the checkpoint with `torch.load(weights_only=True)`. Pass False only for a
            checkpoint you trust whose hyperparameters hold Python objects.

    Your project (`__lighter__.py`) is searched for upward from the checkpoint's folder, then from the current one.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=weights_only)
    if "lighter_config" not in checkpoint:
        raise ValueError(f"'{ckpt_path}' holds no Lighter config: it was not saved by a run that Lighter built.")
    root = _find_up(Path(ckpt_path).resolve().parent, "__lighter__.py") or _find_up(Path.cwd(), "__lighter__.py")
    if root is not None:
        import_module_from_path("project", root)
    model = load_config(checkpoint["lighter_config"], *overrides).resolve("model")
    model.load_state_dict(checkpoint["state_dict"])
    return model
