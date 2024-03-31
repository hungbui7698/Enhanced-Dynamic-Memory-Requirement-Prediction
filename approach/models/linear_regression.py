from functools import cache
from typing import Sequence, Type, Any, Dict

import numpy as np
from overrides import overrides
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from . import TimeModel, ValueModel, BaseModel
from .. import only_assert_if_debug
from ..autoML import ClampedModel, OffsetDirection, OffsetModel
from ..files import TaskTimeseriesCSVFileList, DenoiseParameters


class __LinearRegression(BaseModel):
    name = "LinearRegression"
    intercept_: float
    slope_: float

    @overrides(check_signature=False)
    def train(
        self, x: np.ndarray[float], y: np.ndarray[float], *args, **kwargs
    ) -> "LinearRegression":
        only_assert_if_debug(
            len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[1] == 1)
        )
        only_assert_if_debug(
            len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1)
        )
        x_mean = np.mean(x.ravel())
        y_mean = np.mean(y.ravel())
        Sxx, Sxy = np.cov(x.ravel(), y.ravel(), bias=True)[
            0
        ]  # todo not sure if bias should be set here, apparently this tells numpy to either produce a sample covariance (
        # bias=False) or a population covariance (bias=True)
        self.slope_ = Sxy / Sxx
        self.intercept_ = y_mean - (self.slope_ * x_mean)
        only_assert_if_debug(isinstance(self.slope_, float))
        only_assert_if_debug(isinstance(self.intercept_, float))
        return self

    @overrides(check_signature=False)
    def predict(self, x: float, *args, **kwargs) -> np.ndarray[float]:
        only_assert_if_debug(self.intercept_ is not None and self.slope_ is not None)
        only_assert_if_debug(isinstance(x, (float, np.ndarray)))
        if isinstance(x, np.ndarray):
            only_assert_if_debug(
                len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[1] == 1)
            )
        return np.asarray([self.intercept_ + self.slope_ * x])


class LinearRegressionTimeModel(TimeModel):
    name = "LinearRegressionTimeModel"
    model: ClampedModel

    def __init__(
        self,
        offset_direction: OffsetDirection = OffsetDirection.Down,
        clamp_min: float = 1,
        clamp_max: float = None,
        reg_model_type: Type[BaseEstimator] = None,
        use_preprocessed: bool | DenoiseParameters = False,
        reg_model_params: Dict[str, Any] = None,
    ):
        if reg_model_params is None:
            reg_model_params = {}
        self.offset_direction = offset_direction
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        if reg_model_type is None:
            self.reg_model = LinearRegression()
        else:
            self.reg_model = reg_model_type(**reg_model_params)
        self.use_preprocessed = use_preprocessed
        self.reg_model_params = reg_model_params

    def train(
        self, files: TaskTimeseriesCSVFileList, *args, **kwargs
    ) -> "LinearRegressionTimeModel":
        x = files.input_total_sizes
        y = files.lengths(self.use_preprocessed)
        self.model = ClampedModel(
            OffsetModel(self.reg_model, self.offset_direction).fit(x.reshape(-1, 1), y),
            min_=self.clamp_min,
            max_=self.clamp_max,
        )
        return self

    def predict(self, input_size: float, *args, **kwargs) -> int:
        return self.model.predict(np.array([input_size]).reshape(-1, 1)).astype(int)[0]

    @cache
    def __repr__(self):
        return f"{self.name}({self.offset_direction.name}, {self.clamp_min}, {self.clamp_max}, {self.reg_model!r}, {self.use_preprocessed}, {self.reg_model_params})"


class LinearRegressionValueModel(ValueModel):
    name = "LinearRegressionValueModel"
    model: ClampedModel

    def __init__(
        self,
        offset_direction: OffsetDirection = OffsetDirection.Up,
        clamp_min: float = 1,
        clamp_max: float = None,
        reg_model_type: Type[BaseEstimator] = None,
        reg_model_params: Dict[str, Any] = None,
    ):
        if reg_model_params is None:
            reg_model_params = {}
        self.offset_direction = offset_direction
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        if reg_model_type is None:
            self.reg_model = LinearRegression()
        else:
            self.reg_model = reg_model_type(**reg_model_params)
        self.reg_model_params = reg_model_params

    def train(
        self,
        input_sizes: Sequence[float],
        target_values: Sequence[float],
        *args,
        **kwargs,
    ) -> "LinearRegressionValueModel":
        self.model = ClampedModel(
            OffsetModel(self.reg_model, self.offset_direction).fit(
                np.array(input_sizes).reshape(-1, 1), np.array(target_values)
            ),
            min_=self.clamp_min,
            max_=self.clamp_max,
        )
        return self

    def predict(self, input_size: float, *args, **kwargs) -> float:
        return self.model.predict(np.array([input_size]).reshape(-1, 1)).astype(float)[
            0
        ]

    @cache
    def __repr__(self):
        return f"{self.name}({self.offset_direction.name}, {self.clamp_min}, {self.clamp_max}, {self.reg_model!r}, {self.reg_model_params})"
