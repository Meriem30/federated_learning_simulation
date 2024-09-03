from enum import StrEnum, auto
import copy
from typing import Any, Self

"""
    Two classes merged in one file
"""


class ConfigBase:
    """
        Enable temporary modifications to configuration objects  within a 'with' block
    """
    def __init__(self) -> None:
        self.__old_config: Any = None

    def __enter__(self) -> Self:
        self.__old_config = copy.deepcopy(self)
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        for name in dir(self):
            if not name.startswith("_"):
                setattr(self, name, getattr(self.__old_config, name))
        self.__old_config = None


"""
    For each class:
    Set a unique value for each enumeration member automatically using auto()
    Ensure that the enumeration members are strings using StrEnum => MachineLearningPhase.Training := "Training"
    StrEnum & auto =>  the enumeration members and their values are the same (self-explanatory)
"""


class MachineLearningPhase(StrEnum):
    Training = auto()
    Validation = auto()
    Test = auto()


class EvaluationMode(StrEnum):
    Training = auto()
    Test = auto()
    TestWithGrad = auto()


class ModelType(StrEnum):
    Classification = auto()
    Detection = auto()
    TextGeneration = auto()


class DatasetType(StrEnum):
    Vision = auto()
    Text = auto()
    Graph = auto()
    Audio = auto()
    CodeText = auto()
    Unknown = auto()


class TransformType(StrEnum):
    ExtractData = auto()
    InputText = auto()
    Input = auto()
    RandomInput = auto()
    InputBatch = auto()
    Target = auto()
    TargetBatch = auto()


class ExecutorHookPoint(StrEnum):
    BEFORE_EXECUTE = auto()
    AFTER_EXECUTE = auto()
    BEFORE_EPOCH = auto()
    AFTER_EPOCH = auto()
    MODEL_FORWARD = auto()
    BEFORE_FETCH_BATCH = auto()
    AFTER_FETCH_BATCH = auto()
    BEFORE_BATCH = auto()
    AFTER_BATCH = auto()
    AFTER_VALIDATION = auto()


class StopExecutingException(Exception):
    pass


class IterationUnit(StrEnum):
    Batch = auto()
    Epoch = auto()
    Round = auto()
