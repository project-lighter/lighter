from enum import Enum


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
