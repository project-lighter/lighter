from dataclasses import dataclass
from typing import Dict, Any, Optional
from omegaconf import MISSING


@dataclass
class TrainConfig:
    engine: str = "pytorchito.engines.Trainer"
    epochs: int = MISSING
    batch_size: int = MISSING
    cuda: bool = True
    num_workers: int = 0
    pin_memory: bool = True

    model: Dict = MISSING
    dataset: Dict = MISSING
    transform: Optional[Any] = None  # Union[Dict, List[Dict]]
    target_transform: Optional[Any] = None  # Union[Dict, List[Dict]]
    sampler: Optional[Dict] = None
    optimizer: Any = MISSING
    criteria: Any = MISSING  # Union[Dict, List[Dict]]
    metrics: Optional[Any] = None  # Union[Dict, List[Dict]]


@dataclass
class ValidationConfig:
    engine: str = "pytorchito.engines.Validator"
    dataset: Dict = MISSING
    criteria: Any = MISSING  # Union[Dict, List[Dict]]
    metrics: Optional[Any] = None  # Union[Dict, List[Dict]]


@dataclass
class TestConfig:
    engine: str = "pytorchito.engines.Tester"
    dataset: Dict = MISSING
    metrics: Any = MISSING  # Union[Dict, List[Dict]]
    checkpoint: str = MISSING


@dataclass
class InferenceConfig:
    engine: str = "pytorchito.engines.Inferer"
    checkpoint: str = MISSING


@dataclass
class Config:
    _mode: str = "train"
    project: Optional[str] = None
    task: str = MISSING

    train: Optional[TrainConfig] = None
    val: Optional[ValidationConfig] = None
    test: Optional[TestConfig] = None
    infer: Optional[InferenceConfig] = None
