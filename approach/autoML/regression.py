import enum
import warnings
from enum import Enum
from typing import List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    ARDRegression,
    BayesianRidge,
    HuberRegressor,
    SGDRegressor,
    TheilSenRegressor,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler


class ClampedModel(BaseEstimator):
    base_model = None
    min_ = 0.0
    max_ = 0.0

    def __init__(self, base_model, min_=None, max_=None):
        self.base_model = base_model
        self.min_ = min_
        self.max_ = max_

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "ClampedModel":
        self.base_model.fit(x_train, y_train)
        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        pred = self.base_model.predict(x_test)
        if self.min_ is not None:
            pred = np.fmax(pred, self.min_)
        if self.max_ is not None:
            pred = np.fmin(pred, self.max_)
        return pred


class OffsetDirection(Enum):
    Up = enum.auto()
    Down = enum.auto()
    NOP = enum.auto()


class OffsetModel(BaseEstimator):
    base_model = None
    offset = 0

    def __init__(self, base_model, direction: OffsetDirection = OffsetDirection.Up):
        self.base_model = base_model
        self.direction = direction

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "OffsetModel":
        self.base_model.fit(x_train, y_train)
        if self.direction == OffsetDirection.Up:
            self.offset = max(
                y_train.ravel() - self.base_model.predict(x_train).ravel()
            )
        elif self.direction == OffsetDirection.Down:
            self.offset = min(
                y_train.ravel() - self.base_model.predict(x_train).ravel()
            )
        elif self.direction == OffsetDirection.NOP:
            self.offset = 0
        else:
            raise ValueError("Wrong offset direction")
        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        return self.base_model.predict(x_test) + self.offset

    def score(self, x_test, y_test):
        if len(x_test) > 1:
            return r2_score(y_test, self.predict(x_test))
        return 1  # TODO this silences warning that R2 is not defined for single point test sets -- dont know how we're getting single data tests though


class Regression(BaseEstimator):
    candidates: List[OffsetModel] = None
    model: OffsetModel | None = None
    scaler: StandardScaler | None = None

    def __init__(self):
        self.model = None
        self.candidates = [
            OffsetModel(LinearRegression()),
            OffsetModel(Ridge()),
            OffsetModel(SGDRegressor()),
            OffsetModel(ARDRegression()),
            OffsetModel(BayesianRidge()),
            OffsetModel(HuberRegressor()),
            OffsetModel(TheilSenRegressor()),
            # OffsetModel(PoissonRegressor()),
            # OffsetModel(TweedieRegressor()),
            # OffsetModel(GammaRegressor()),
        ]
        self.scaler = None
        warnings.simplefilter("ignore", DataConversionWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)

    def fit(self, x_train, y_train) -> "Regression":
        x_ = np.array(x_train).reshape(-1, 1)
        y_ = np.array(y_train).reshape(-1, 1)
        res = []
        x_ = np.asarray([x_, x_**2, x_**3, np.log(x_) * x_]).reshape(4, -1).T
        scaler = StandardScaler().fit(x_)
        self.scaler = scaler
        x_ = scaler.transform(x_)
        for c in self.candidates:
            cv = cross_validate(c, x_, y_, return_train_score=True)
            te = np.asarray(cv["test_score"])
            tr = np.asarray(cv["train_score"])
            # sc = te * tr
            res.append(np.mean(te))
            c.fit(x_, y_)
            # res.append(c.score(x_, y_))
        self.model = self.candidates[np.argmax(res)]
        # rc.log(self.model)
        return self

    def predict(self, x_test) -> np.ndarray:
        x_ = np.asarray(x_test).reshape(-1, 1)
        x_ = self.scaler.transform(
            np.asarray([x_, x_**2, x_**3, np.log(x_) * x_]).reshape(4, -1).T
        )
        return np.array(self.model.predict(x_)).ravel()

    def regress(self, x_train, y_train, x_test) -> np.ndarray:
        return self.fit(x_train, y_train).predict(x_test)
