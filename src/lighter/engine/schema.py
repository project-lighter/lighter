"""
Defines the schema for configuration validation using Sparkwheel's validation with dataclasses.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AdapterConfig:
    """Adapter configuration for a specific mode."""

    batch: Optional[dict[str, Any]] = None
    criterion: Optional[dict[str, Any]] = None
    metrics: Optional[dict[str, Any]] = None
    logging: Optional[dict[str, Any]] = None


@dataclass
class PredictAdapterConfig:
    """Adapter configuration for predict mode (no criterion)."""

    batch: Optional[dict[str, Any]] = None
    logging: Optional[dict[str, Any]] = None


@dataclass
class AdaptersConfig:
    """Adapters configuration for all modes."""

    train: Optional[dict[str, Any] | str] = None  # Can be AdapterConfig but keep flexible
    val: Optional[dict[str, Any] | str] = None
    test: Optional[dict[str, Any] | str] = None
    predict: Optional[dict[str, Any] | str] = None


@dataclass
class MetricsConfig:
    """Metrics configuration for different stages."""

    train: Optional[list[Any] | dict[str, Any] | str] = None
    val: Optional[list[Any] | dict[str, Any] | str] = None
    test: Optional[list[Any] | dict[str, Any] | str] = None


@dataclass
class DataloadersConfig:
    """Dataloaders configuration for different stages."""

    train: Optional[dict[str, Any]] = None
    val: Optional[dict[str, Any]] = None
    test: Optional[dict[str, Any]] = None
    predict: Optional[dict[str, Any]] = None


@dataclass
class SystemConfig:
    """System configuration with model, optimizer, scheduler, etc."""

    model: Optional[dict[str, Any]] = None
    criterion: Optional[dict[str, Any]] = None
    optimizer: Optional[dict[str, Any]] = None
    scheduler: Optional[dict[str, Any]] = None
    inferer: Optional[dict[str, Any]] = None
    metrics: Optional[MetricsConfig] = None
    dataloaders: Optional[DataloadersConfig] = None
    adapters: Optional[AdaptersConfig] = None


@dataclass
class ArgsConfig:
    """Arguments to pass to Trainer stage methods."""

    fit: Optional[dict[str, Any]] = None
    validate: Optional[dict[str, Any]] = None
    test: Optional[dict[str, Any]] = None
    predict: Optional[dict[str, Any]] = None


@dataclass
class ConfigSchema:
    """Main Lighter configuration schema."""

    trainer: dict[str, Any]  # pytorch_lightning.Trainer
    system: SystemConfig  # lighter.System
    project: Optional[str] = None
    vars: Optional[dict[str, Any]] = None
    args: Optional[ArgsConfig] = None
