import enum
import warnings
from abc import ABC
from enum import Enum
from functools import cache
from typing import TypeVar, Sequence

import numpy as np
from overrides import overrides

from .. import MissingReprWarning
from ..files import TaskTimeseriesCSVFileList, TaskTimeseriesCSVFiles, DenoiseParameters

T = TypeVar("T")


class BaseModel(ABC):
    name: str = "BaseModel"
    use_preprocessed: bool | DenoiseParameters = False

    def train(self: T, files: TaskTimeseriesCSVFileList, *args, **kwargs) -> T:
        return self

    def predict(self, x: float, *args, **kwargs) -> np.ndarray[float]:
        pass

    @cache
    def __repr__(self):
        warnings.warn(
            "Remember to implement a __repr__ on all models that includes parameter information of the instance so that the plots will be able to differentiate them",
            MissingReprWarning,
        )
        return self.name


class TimeModel(BaseModel):
    name = "TimeModel"

    # models the expected duration of a task
    @overrides(check_signature=False)
    def predict(self, input_size: float, *args, **kwargs) -> int:
        pass


class ValueModel(
    BaseModel
):  # todo remove the inheritance from BaseModel on classes that override the signatures/functions anyway..?
    name = "ValueModel"

    # models the expected usage for a segment of a tasks runtime
    @overrides(check_signature=False)
    def train(
        self, input_sizes: Sequence[float], maxes: Sequence[float], *args, **kwargs
    ) -> "ValueModel":
        return self

    @overrides(check_signature=False)
    def predict(self, input_size: float, *args, **kwargs) -> float:
        pass


class RetryMode(Enum):
    Full = enum.auto()
    Selective = enum.auto()
    Partial = enum.auto()
    Tovar = enum.auto()


class RetryModel(BaseModel):
    name = "RetryModel"

    # model/function that predicts the retry factor to use
    @overrides(check_signature=False)
    def predict(
        self, usage: float, allocation: float, retry_mode: RetryMode, *args, **kwargs
    ) -> float:
        # Note: RetryModels should always predict at least usage/allocation, so that only one retry is needed to fix each failure!
        pass


class SegmentModel(BaseModel):
    name = "SegmentModel"
    num_segments: int

    # models the expected locations of splits between segments of a tasks runtime
    # NOTE: segments are run length encoded by default i.e. SegmentModel.predict gives something like [3,3,3,4]
    @overrides(check_signature=False)
    def predict(
        self,
        runtime: int,
        input_size: float,
        oracle: TaskTimeseriesCSVFiles | None,
        *args,
        **kwargs,
    ) -> np.ndarray[int]:
        # predict locations (i.e. run lengths of segments) of change points (segment boundaries) based on runtime
        # oracle is provided during training, but unavailable while testing
        pass


class TaskModel(BaseModel):
    name: str = "TaskModel"

    # models the usage of a task over its whole runtime
    pass
