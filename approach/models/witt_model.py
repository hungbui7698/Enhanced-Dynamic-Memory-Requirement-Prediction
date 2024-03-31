import enum
from enum import Enum
from functools import cache
from typing import Sequence

import numpy as np
from sklearn.linear_model import LinearRegression

from . import TaskModel
from .. import only_assert_if_debug
from ..files import TaskTimeseriesCSVFileList, DenoiseParameters


def train_lm(x: Sequence[float], y: Sequence[float]) -> LinearRegression:
    return LinearRegression().fit(np.asarray(x).reshape((-1, 1)), y)


def lr_max_shift(
    x: Sequence[float], y: Sequence[float], model: LinearRegression
) -> float:
    l_diff = 0
    for i, v in enumerate(x):
        pred = model.predict(np.array([v]).reshape((-1, 1)))
        diff = y[i] - pred
        l_diff = min(diff, l_diff)
    return abs(l_diff)


def lr_mean_shift(
    x: Sequence[float], y: Sequence[float], model: LinearRegression
) -> float:
    s = 0
    for i, v in enumerate(x):
        f_x = model.predict(np.array([v]).reshape((-1, 1)))
        s += (f_x - y[i]) ** 2
    return np.sqrt(
        s / (max(1, len(y) - 1))
    )  # todo this actually uses len(y) - 1 in nils code .. but that breaks if there is only one training file


def lr_mean_negative_shift(
    x: Sequence[float], y: Sequence[float], model: LinearRegression
) -> float:
    s = 0
    c = 0
    for i, v in enumerate(x):
        f_x = model.predict(np.array([v]).reshape((-1, 1)))
        if f_x < y[i]:
            c += 1
            s += (f_x - y[i]) ** 2
    c = max(2, c)
    return np.sqrt(s / (c - 1))


class WittMode(Enum):
    Mean = enum.auto()
    MeanNegative = enum.auto()
    Max = enum.auto()


WittMode.mode_intercept_fn_lut = {
    WittMode.Mean: lr_mean_shift,
    WittMode.MeanNegative: lr_mean_negative_shift,
    WittMode.Max: lr_max_shift,
}


class WittModel(TaskModel):
    name = "WittModel"
    mode: WittMode
    default_min: float
    model: LinearRegression

    def __init__(
        self,
        mode: WittMode = WittMode.Mean,
        default_min=100.0,
        use_preprocessed: bool | DenoiseParameters = False,
    ):
        self.default_min = default_min
        self.mode = mode
        self.model = None
        self.use_preprocessed = use_preprocessed

    def predict(self, x: float, *args, **kwargs) -> np.ndarray[float]:
        only_assert_if_debug(self.model is not None)
        ram = self.model.predict(np.array([x]).reshape((-1, 1))).astype(float)[0]
        if ram <= 0:
            ram = self.default_min
        return np.array([ram])

    def train(
        self, files: TaskTimeseriesCSVFileList = None, *args, **kwargs
    ) -> "WittModel":
        only_assert_if_debug(files is not None or ("x" in kwargs and "y" in kwargs))
        if files is None:
            x = kwargs["x"]
            y = kwargs["y"]
        else:
            x = files.input_total_sizes
            y = files.peak_usages(self.use_preprocessed)

        lm = train_lm(x, y)
        lm.intercept_ += WittMode.mode_intercept_fn_lut[self.mode](x, y, lm)
        self.model = lm
        return self

    @cache
    def __repr__(self):
        return f"{self.name}({self.mode.name}, {self.use_preprocessed})"
