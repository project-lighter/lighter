from enum import Enum


class StrEnum(str, Enum):
    """
    Enum class that inherits from str. This allows for the enum values to be accessed as strings.
    """

    # Remove this class when Python 3.10 support is dropped, as Python >=3.11 has StrEnum built-in.
    def __str__(self) -> str:
        return str(self.value)


class Data(StrEnum):
    IDENTIFIER = "identifier"
    INPUT = "input"
    TARGET = "target"
    PRED = "pred"
    LOSS = "loss"
    METRICS = "metrics"
    STEP = "step"
    EPOCH = "epoch"


class Stage(StrEnum):
    FIT = "fit"
    VALIDATE = "validate"
    TEST = "test"
    PREDICT = "predict"
    LR_FIND = "lr_find"
    SCALE_BATCH_SIZE = "scale_batch_size"


class Mode(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    PREDICT = "predict"
