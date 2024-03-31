import functools
from numbers import Number
from typing import List

import numpy as np

from approach import only_assert_if_debug


def max_around(arr: np.ndarray | List[Number], i: int, stride=15):
    only_assert_if_debug(0 <= i <= len(arr) - 1, "i must be a valid index")
    only_assert_if_debug(stride % 2 != 0, "Must be odd")
    s_ = (stride - 1) // 2
    return max(arr[max(i - s_, 0) : min(i + s_ + 1, len(arr))])


def lerp(a: Number, b: Number, t: Number) -> Number:
    only_assert_if_debug(0 <= t <= 1)
    return a + (b - a) * t


def many_max(_2darr: np.ndarray) -> np.ndarray:
    return np.max(_2darr, axis=0)


def add_to_uneven(num: Number) -> int:
    if num % 2 == 0:
        return num + 1
    return num


def uneven_div2(num: Number) -> int:
    r = num // 2
    return add_to_uneven(r)


class Interval:  # should maybe be called a span since it includes both endpoints in inclusion comparisons
    low: float
    high: float
    steps: int | None

    def __init__(self, a: float, b: float, s: int | None = None):
        self.low = a
        self.high = b
        self.steps = s

    @property
    @functools.cache
    def items(self):
        o = np.linspace(self.low, self.high, self.steps + 2, endpoint=True)
        only_assert_if_debug(
            (o.max() <= self.high and self.low <= o.min()),
            f"Generated between ({o.min()}, {o.max()}), but limits ({self.low}, {self.high})",
        )
        return o

    def __contains__(self, item):
        if isinstance(item, Number):
            if self.steps is None:
                return self.low <= item <= self.high
            else:
                return item in self.items
        elif isinstance(item, Interval):
            return self.low <= item.low and item.high <= self.high
        else:
            raise RuntimeError("Incomparable item was passed")

    def __repr__(self):
        return f"({self.low}, {self.high})"
