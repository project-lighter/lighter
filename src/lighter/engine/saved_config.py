"""Save each run's composed config, and rebuild a model from the config saved in its checkpoint."""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import yaml
from pytorch_lightning import Callback

from lighter.engine.runner import _find_up, load_config, to_yaml
from lighter.utils.dynamic_imports import import_module_from_path

logger = logging.getLogger(__name__)


class SaveConfig(Callback):
    """Added to every trainer Lighter builds: saves the composed config with the run.

    On the global-zero process it writes `config.yaml` to the log folder and logs the config as
    hyperparameters, once the log folder is final (also under DDP). Every checkpoint stores the config
    under `"lighter_config"`, which `load_model` reads.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:
        log_dir = trainer.log_dir if trainer.logger else None  # every rank asks: under DDP it is a broadcast
        if trainer.is_global_zero and trainer.logger:
            trainer.logger.log_hyperparams(self.config)
            if log_dir:
                try:
                    text = to_yaml(self.config)
                except yaml.YAMLError:  # a dict config holding Python objects
                    logger.warning("config.yaml is not saved: the config holds Python objects that YAML cannot write.")
                    return
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                (Path(log_dir) / "config.yaml").write_text(text)

    def on_save_checkpoint(self, trainer: Any, pl_module: Any, checkpoint: dict) -> None:
        checkpoint["lighter_config"] = self.config


def load_model(ckpt_path: str | Path, overrides: Iterable[str] = (), weights_only: bool = True) -> Any:
    """Rebuild the model of a Lighter run from the config in its checkpoint, and load the checkpoint's weights.

    Args:
        ckpt_path: A checkpoint saved by a run that Lighter built.
        overrides: Overrides applied to the saved config first, e.g. `["model::network::dropout=0"]`.
        weights_only: Load the checkpoint with `torch.load(weights_only=True)`. Pass False only for a
            checkpoint you trust whose hyperparameters hold Python objects.

    A checkpoint saved with `ModelCheckpoint(save_weights_only=True)` holds no config, so the run's
    `config.yaml` next to its `checkpoints` folder is used. Your project (`__lighter__.py`) is searched for
    upward from the checkpoint's folder, then from the current one.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=weights_only)
    config = checkpoint.get("lighter_config") or Path(ckpt_path).resolve().parents[1] / "config.yaml"
    if isinstance(config, Path) and not config.is_file():
        raise ValueError(
            f"'{ckpt_path}' holds no Lighter config (checkpoints saved with `save_weights_only=True` don't), and "
            f"there is no {config}. Build the model with `lighter.build(<config>)` and load the weights yourself."
        )
    root = _find_up(Path(ckpt_path).resolve().parent, "__lighter__.py") or _find_up(Path.cwd(), "__lighter__.py")
    if root is not None:
        import_module_from_path("project", root)
    model = load_config(config, *overrides).resolve("model")
    model.load_state_dict(checkpoint["state_dict"])
    return model
