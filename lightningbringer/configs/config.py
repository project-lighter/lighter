from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
from lightningbringer import System
from lightningbringer.configs.utils import generate_omegaconf_dataclass

TrainerConfig = generate_omegaconf_dataclass("TrainerConfig", pl.Trainer)
SystemConfig = generate_omegaconf_dataclass("SystemConfig", System)


@dataclass
class Config:
    project: Optional[str] = None
    trainer: TrainerConfig = TrainerConfig()
    system: SystemConfig = SystemConfig()
