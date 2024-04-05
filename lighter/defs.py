from enum import StrEnum


class ModeEnum(StrEnum):
    """Enum for the modes of operation."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    PREDICT = "predict"
