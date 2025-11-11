"""
Defines the schema for configuration validation using Sparkwheel's validation with dataclasses.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AdapterConfig:
    """Adapter configuration for a specific mode."""

    batch: Optional[dict] = None
    criterion: Optional[dict] = None
    metrics: Optional[dict] = None
    logging: Optional[dict] = None


@dataclass
class PredictAdapterConfig:
    """Adapter configuration for predict mode (no criterion)."""

    batch: Optional[dict] = None
    logging: Optional[dict] = None


@dataclass
class AdaptersConfig:
    """Adapters configuration for all modes."""

    train: Optional[dict] = None  # Can be AdapterConfig but keep flexible
    val: Optional[dict] = None
    test: Optional[dict] = None
    predict: Optional[dict] = None


@dataclass
class MetricsConfig:
    """Metrics configuration for different stages."""

    train: Optional[list | dict] = None
    val: Optional[list | dict] = None
    test: Optional[list | dict] = None


@dataclass
class DataloadersConfig:
    """Dataloaders configuration for different stages."""

    train: Optional[dict] = None
    val: Optional[dict] = None
    test: Optional[dict] = None
    predict: Optional[dict] = None


@dataclass
class SystemConfig:
    """System configuration with model, optimizer, scheduler, etc."""

    model: Optional[dict] = None
    criterion: Optional[dict] = None
    optimizer: Optional[dict] = None
    scheduler: Optional[dict] = None
    inferer: Optional[dict] = None
    metrics: Optional[MetricsConfig] = None
    dataloaders: Optional[DataloadersConfig] = None
    adapters: Optional[AdaptersConfig] = None


@dataclass
class ArgsConfig:
    """Arguments to pass to Trainer stage methods."""

    fit: Optional[dict] = None
    validate: Optional[dict] = None
    test: Optional[dict] = None
    predict: Optional[dict] = None


@dataclass
class LighterConfig:
    """Main Lighter configuration schema."""

    trainer: dict  # pytorch_lightning.Trainer
    system: SystemConfig  # lighter.System
    project: Optional[str] = None
    vars: Optional[dict] = None
    args: Optional[ArgsConfig] = None
