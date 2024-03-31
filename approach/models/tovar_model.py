import warnings
from functools import cache

import numpy as np

from . import TaskModel
from .FirstAllocation import FirstAllocation
from .. import only_assert_if_debug
from ..files import TaskTimeseriesCSVFileList, DenoiseParameters


class TovarModel(TaskModel):
    name = "TovarModel"
    fa: FirstAllocation

    def __init__(self, use_preprocessed: bool | DenoiseParameters = False):
        self.fa = None
        self.use_preprocessed = use_preprocessed

    def train(
        self, files: TaskTimeseriesCSVFileList = None, *args, **kwargs
    ) -> "TovarModel":
        only_assert_if_debug(
            files is not None or ("maxes" in kwargs and "times" in kwargs)
        )
        if files is not None:
            maxes = files.peak_usages(self.use_preprocessed)
            times = files.lengths(self.use_preprocessed)
        else:
            warnings.warn(
                "Better not specify maxes and times explicitly unless you're sure you know what you are doing!",
                RuntimeWarning,
            )
            maxes = kwargs["maxes"]
            times = kwargs["times"]
        self.fa = FirstAllocation(name="my memory usage")
        for max_, time in zip(maxes, times):
            self.fa.add_data_point(value=max_, time=time)
        return self

    def predict(self, x: np.ndarray, *args, **kwargs) -> np.ndarray[float]:
        # todo this doesn't use parameter x..? seems like this is just how tovar operates..?
        return np.array([max(1.0, self.fa.first_allocation(mode="waste"))])

    @cache
    def __repr__(self):
        return f"{self.name}({self.use_preprocessed})"
