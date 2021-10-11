from dataclasses import dataclass
from typing import Any, Dict, Optional

from omegaconf import MISSING, OmegaConf

import pytorch_lightning as pl

from pytorchito import System
from pytorchito.configs.utils import generate_omegaconf_dataclass

OmegaConf.register_new_resolver("get_model_parameters", lambda x: x.parameters())

TrainerConfig = generate_omegaconf_dataclass("TrainerConfig", pl.Trainer)
SystemConfig = generate_omegaconf_dataclass("SystemConfig", System)

@dataclass
class Config:
    project: Optional[str] = None

    trainer: TrainerConfig = TrainerConfig()
    system: SystemConfig = SystemConfig()
