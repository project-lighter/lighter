from enum import StrEnum


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
