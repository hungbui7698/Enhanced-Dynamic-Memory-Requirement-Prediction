import itertools
import warnings
from enum import Enum, auto
from functools import partial
from numbers import Number
from typing import List, Iterable, Sequence

import numpy as np

from . import basic
from .. import only_assert_if_debug


def blockify(arr: np.ndarray | List[Number], stride: int = 15) -> np.ndarray:
    """
    A rolling maximum centered around each entry in @arr with total width @stride
    :param arr: input array
    :param stride: total amount of entries in the rolling maximum including the center entry itself -- must be odd
    :return: an array of length len(arr) of the rolling maximum over @arr
    """
    if isinstance(arr, np.ndarray):
        only_assert_if_debug(len(arr.shape) == 1 or arr.shape[1] == 1)
    return np.array([basic.max_around(arr, i, stride) for i in range(len(arr))])


def stepify(
    arr: np.ndarray | List[Number], stride: int | List[int] = 15, is_num: bool = False
) -> np.ndarray:
    """
    A walking maximum in @arr with total width @stride or @stride amount of steps if @is_num is True
    :param arr: input array
    :param stride: total amount of entries in the maximum, number of maximums of @is_num, or a list of break point indices where maximums should start
    :param is_num: whether stride represents an amount of steps to output or is the width of the maximums
    :return: an array of length len(arr) of the maximums over @arr
    """
    if isinstance(arr, np.ndarray):
        only_assert_if_debug(len(arr.shape) == 1 or arr.shape[1] == 1)
    if isinstance(stride, Number):
        if is_num:
            _stride = int(np.ceil(len(arr) / stride))
        else:
            _stride = stride
        maxes = (
            max(arr[i * _stride : min((i + 1) * _stride, len(arr))])
            for i in range((len(arr) // _stride) + (len(arr) % _stride > 0))
        )
        maxes = [x for m in maxes for x in [m] * _stride]
    elif isinstance(stride, (np.ndarray, List)):
        c = 0
        lens = []
        maxes = []
        _stride = [x for x in stride if 0 < x < len(arr)] + [len(arr)]
        for p in _stride:
            maxes.append(max(arr[c:p]))
            lens.append(p - c)
            c = p
        maxes = [x for m, l in zip(maxes, lens) for x in [m] * l]
    else:
        raise RuntimeError("Wrong type of stride")
    return np.array(maxes)[: len(arr)]


def smooth(
    arr: np.ndarray | List[Number], width: int = 15, freeze_ends: bool = False
) -> np.ndarray:
    """
    Apply an averaging convolution filter of width @width to the input @arr, leaving the first and last entries unchanged if @freeze_ends
    :param arr: input array
    :param width: width of the smoothing filter including the center entry itself -- must be odd
    :param freeze_ends: whether to leave the first and last entries unchanged
    :return: array of the smoothed values
    """
    if isinstance(arr, np.ndarray):
        only_assert_if_debug(len(arr.shape) == 1 or arr.shape[1] == 1)
    # only_assert_if_debug(width % 2 == 1, "not sure how to deal with even width yet")
    if width % 2 == 0:
        width = width + 1
    buflen = int((width - 1) / 2)
    only_assert_if_debug(buflen == np.floor((width - 1) / 2), "should be an int")
    if freeze_ends:
        a, b = arr[0], arr[-1]
    o = np.convolve(
        [*([arr[0]] * buflen), *arr, *([arr[-1]] * buflen)],
        [1 / width] * width,
        mode="valid",
    )
    if freeze_ends:
        o[0], o[-1] = a, b
    return o


def smooth_more(
    arr: np.ndarray | List[Number],
    widths: int | Iterable[Number] = 15,
    depth: int = 2,
    freeze_ends: bool = False,
) -> np.ndarray:
    """
    Repeatedly apply an averaging convolution filter of width @widths or @widths[i] to the input @arr, leaving the first and last entries unchanged if @freeze_ends.
    If widths is an Iterable, then the values in it will be cycled through on each iteration.
    :param arr: input array
    :param widths: width of the smoothing filter including the center entry itself -- must be odd -- or an Iterable of such values
    :param depth: amount of iterations to perform
    :param freeze_ends: whether to leave the first and last entries unchanged
    :return: array of the smoothed values
    """
    if isinstance(arr, np.ndarray):
        only_assert_if_debug(len(arr.shape) == 1 or arr.shape[1] == 1)
    if not isinstance(widths, Iterable):
        only_assert_if_debug((widths % 2) != 0, "smooth doesnt handle even widths yet")
        w_ = itertools.cycle([widths])
    else:
        only_assert_if_debug(
            all((np.array(widths) % 2) != 0), "smooth doesnt handle even widths yet"
        )
        if len(list(itertools.islice(widths, depth))) < depth:
            warnings.warn("will cycle widths", RuntimeWarning)
        w_ = itertools.cycle(widths)
    c_ = smooth(arr, next(w_), freeze_ends=freeze_ends)
    for d, w in zip(range(depth - 1), w_):
        c_ = smooth(c_, w, freeze_ends=freeze_ends)
    return c_


class MaxerTypes(Enum):
    blockify = auto()
    stepify = auto()


def denoise(
    arr: np.ndarray | List[Number],
    stride: int = 15,
    freeze_ends: bool = False,
    maxer_type: MaxerTypes = MaxerTypes.blockify,
):
    """
    Apply default denoising by smoothing with half of the stride applied while blockifying
    :param arr: input array
    :param stride: size to use while blockifying -- also determines width of the smoothing
    :param freeze_ends: whether to leave the first and last entries unchanged
    :return: array of the denoised values
    """
    if maxer_type == MaxerTypes.blockify:
        maxer = partial(blockify, stride=stride)
    elif maxer_type == MaxerTypes.stepify:
        maxer = partial(stepify, stride=round(stride / 2))
    else:
        raise ValueError("maxer_type must be from MaxerTypes enum")
    return smooth(maxer(arr), basic.uneven_div2(stride), freeze_ends)


def denoise_more(
    arr: np.ndarray | List[Number],
    stride: int = 15,
    widths: int | Iterable[Number] = None,
    depth: int = None,
    freeze_ends: bool = False,
    maxer_type: MaxerTypes = MaxerTypes.blockify,
) -> np.ndarray:
    """
    Apply default denoising with multiple smoothing steps. If no widths and depth are specified the smoothings will repeatedly halve their width.
    :param arr: input array
    :param stride: size to use while blockifying -- also determines width of the smoothings if no widths are given
    :param widths: the widths to use while smoothing
    :param depth: the amount of smoothings to perform
    :param freeze_ends: whether to leave the first and last entries unchanged
    :return: array of the denoised values
    """
    # assert (widths is None and depth is None) or (
    #     widths is not None and depth is not None
    # ), "Specify either both or none of widths and depth"
    if stride % 2 == 0:
        stride = stride + 1
    if widths is None:
        c_ = basic.uneven_div2(stride)
        w_ = [c_]
        d_ = 1
        while True:
            if depth is not None and d_ >= depth:
                break
            c_ = basic.uneven_div2(c_)
            if c_ <= 1:
                break
            w_.append(c_)
            d_ += 1
        d_ = len(w_)
        only_assert_if_debug(depth is None or d_ == depth)
    else:
        w_ = widths
        d_ = depth
    if maxer_type == MaxerTypes.blockify:
        maxer = partial(blockify, stride=stride)
    elif maxer_type == MaxerTypes.stepify:
        maxer = partial(stepify, stride=round(stride / 2))
    else:
        raise ValueError("maxer_type must be from MaxerTypes enum")
    return smooth_more(maxer(arr), w_, d_, freeze_ends)


class Interpoler:
    """
    Applies linear interpolation between given X and Y data to allow querying arbitrary X values by producing an approximate Y value based on given data
    """

    x_data: np.ndarray
    y_data: np.ndarray

    def __init__(
        self, x: np.ndarray | Sequence[Number], y: np.ndarray | Sequence[Number]
    ):
        try:
            len(x)
            len(y)
        except:
            raise RuntimeError("x and y need to be Sequences")
        only_assert_if_debug(len(x) == len(y))
        self.x_data = x
        if not isinstance(self.x_data, np.ndarray):
            try:
                self.x_data = np.array(self.x_data)
            except:
                raise RuntimeError(
                    "Need to be able to make passed x into a numpy array"
                )
        # self.x_data = self.x_data.reshape(-1, 1)
        self.x_data = self.x_data.ravel()
        self.y_data = y
        if not isinstance(self.y_data, np.ndarray):
            try:
                self.y_data = np.array(self.y_data)
            except:
                raise RuntimeError(
                    "Need to be able to make passed y into a numpy array"
                )
        # self.y_data = self.y_data.reshape(-1, 1)
        self.y_data = self.y_data.ravel()

    def _interpolate_one(self, item):
        only_assert_if_debug(isinstance(item, Number))
        if item in self.x_data:
            return self.y_data[np.where(self.x_data == item)[0][0]]
        if item < self.x_data.min() or item > self.x_data.max():
            raise NotImplementedError(
                f"{item} outside of ({self.x_data.min()},{self.x_data.max()})"
            )
        dist = np.abs(self.x_data - item)
        min_id = dist.argmin()
        if item < self.x_data[min_id]:
            partner_id = min_id - 1
            min_id, partner_id = partner_id, min_id
        elif item > self.x_data[min_id]:
            partner_id = min_id + 1
        else:
            raise RuntimeError("How?")
        a, b = self.y_data[min_id], self.y_data[partner_id]
        c, d = self.x_data[min_id], self.x_data[partner_id]
        return basic.lerp(a, b, (item - c) / (d - c))

    def __getitem__(self, item: Number | Iterable[Number]):
        try:
            iter(item)
        except:
            return self._interpolate_one(item)
        else:
            return [self._interpolate_one(i) for i in item]

    def __repr__(self):
        return f"X\t{self.x_data}\nY\t{self.y_data}"
