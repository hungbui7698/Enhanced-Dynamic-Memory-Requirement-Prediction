from functools import cache
from typing import List

import numpy as np

from . import (
    TaskModel,
    RetryModel,
    RetryMode,
    WittMode,
    WittModel,
    TovarModel,
    T,
)
from .k_equal_segmenting_model import KEqualSegmentingModel
from ..files import TaskTimeseriesCSVFileList


class DefaultTaskModel(TaskModel):  # Does nothing
    name = "DefaultModel"
    pass


class NoCheatDefaultTaskModel(TaskModel):
    name = "NoCheatDefaultModel"

    def __init__(self):
        self.observed_max = 0

    def train(
        self, files: TaskTimeseriesCSVFileList, *args, **kwargs
    ) -> "NoCheatDefaultTaskModel":
        self.observed_max = max(files.peak_usages())
        return self

    def predict(self, x: float, *args, **kwargs) -> np.ndarray[float]:
        return np.asarray([self.observed_max], float)


class DefaultRetryModel(RetryModel):
    name = "DefaultRetryModel"

    def predict(
        self, usage: float, allocation: float, retry_mode: RetryMode, *args, **kwargs
    ) -> float:
        return max(2.0, usage / allocation)

    @cache
    def __repr__(self):
        return self.name


def get_standard_models() -> List[TaskModel]:
    return [
        WittModel(WittMode.Mean),
        WittModel(WittMode.Max),
        WittModel(WittMode.MeanNegative),
        TovarModel(),
        KEqualSegmentingModel(4, None, False, {}),  # Nils KEqual
    ]
