from typing import Any

from dataclasses import dataclass
from enum import Enum


@dataclass
class Batch:
    input: Any
    target: Any | None = None
    id: Any | None = None


class Data(str, Enum):
    ID = "id"
    INPUT = "input"
    TARGET = "target"
    PRED = "pred"


class Stage(str, Enum):
    FIT = "fit"
    VALIDATE = "validate"
    TEST = "test"
    PREDICT = "predict"
    LR_FIND = "lr_find"
    SCALE_BATCH_SIZE = "scale_batch_size"


class Mode(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    PREDICT = "predict"
