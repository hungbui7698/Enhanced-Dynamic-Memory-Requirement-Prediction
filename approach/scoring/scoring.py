from numbers import Number
from typing import List

import numpy as np

from approach import only_assert_if_debug


def delta_measure(
    allocation: np.ndarray | List[Number],
    usage: np.ndarray | List[Number],
    pad_end: bool = False,
) -> float:
    if not pad_end:
        only_assert_if_debug(len(allocation) == len(usage))
        a_ = allocation
    else:
        a_ = np.ones_like(usage) * allocation[-1]
        a_[: len(allocation)] = allocation
    if not isinstance(a_, np.ndarray):
        a_ = np.array(a_)
    if not isinstance(usage, np.ndarray):
        b_ = np.array(usage)
    else:
        b_ = usage
    which_over = a_ >= b_
    over = a_[which_over] - b_[which_over]
    under = b_[np.logical_not(which_over)]
    return sum(over) + sum(under)


def delta_score(
    allocation: np.ndarray | List[Number], usage: np.ndarray | List[Number]
) -> float:
    return delta_measure(allocation, usage) / len(allocation)
