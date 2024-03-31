import enum
import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Callable

from approach.preprocessing import MaxerTypes


class RuptureSegmentAlgorithms(Enum):
    binseg = enum.auto()
    bottom_up = enum.auto()
    dynp = enum.auto()
    kernel_cpd = enum.auto()
    pelt = enum.auto()
    window = enum.auto()


@dataclass(frozen=True)
class DenoiseParameters:  # todo make ABC?
    stride: int | Callable[[...], int] = 15
    widths: int | Iterable[int] = None
    depth: int = None
    freeze_ends: bool = False
    maxer_type: MaxerTypes = MaxerTypes.blockify

    def train(self, *args, **kwargs) -> "DenoiseParameters":
        return self


class StaticDenoiseParameters(DenoiseParameters):
    stride: int = 15

    def __init__(
        self,
        stride: int = 15,
        widths: int | Iterable[int] = None,
        depth: int = None,
        freeze_ends: bool = False,
        maxer_type: MaxerTypes = MaxerTypes.blockify,
    ):
        super().__init__(stride, widths, depth, freeze_ends, maxer_type)

    def train(self, *args, **kwargs) -> "StaticDenoiseParameters":
        return self


def basic_dynamic_stride(
    k: int,
    file: "TaskTimeseriesCSVFiles",
    resolution: int = 1,
    min_: int = 0,
    max_: int = None,
) -> int:
    # produce stride that will split the runtime of the given file into k * resolution parts during preprocessing
    # k should usually be the amount of segments the segment model predicts (if this is fixed)
    # resolution may increase precision of the preprocessing around the edges of segments but tweaking widths of the preprocessing as well may be necessary (especially for high
    #   values of resolution)
    stride = max(2 * math.ceil(len(file) / (k * resolution)), min_)
    if max_ is not None:
        stride = min(stride, max_)
    return stride


class DynamicDenoiseParameters(DenoiseParameters):
    stride: Callable[[...], int] = lambda *x, **y: 15

    def __init__(
        self,
        stride: Callable[[...], int] = lambda *x, **y: 15,
        widths: int | Iterable[int] = None,
        depth: int = None,
        freeze_ends: bool = False,
        maxer_type: MaxerTypes = MaxerTypes.blockify,
    ):
        super().__init__(stride, widths, depth, freeze_ends, maxer_type)

    def train(self, *args, **kwargs) -> StaticDenoiseParameters:
        return StaticDenoiseParameters(
            self.stride(*args, **kwargs),
            self.widths,
            self.depth,
            self.freeze_ends,
            self.maxer_type,
        )


@dataclass
class RuptureParams:
    num_segments: int = None
    start: int = 0
    end: int = None
    model: str = None
    min_size: int = 2
    jump: int = 5
    width: int = 100
    pen: float = None
    epsilon: float = None


class RuptureAlgorithmModelTypes:
    by_algorithm = {
        RuptureSegmentAlgorithms.binseg: ["l1", "l2", "rbf"],
        RuptureSegmentAlgorithms.bottom_up: ["l1", "l2", "rbf"],
        RuptureSegmentAlgorithms.dynp: ["l1", "l2", "rbf"],
        RuptureSegmentAlgorithms.kernel_cpd: ["linear", "rbf", "cosine"],
        RuptureSegmentAlgorithms.pelt: ["l1", "l2", "rbf"],
        RuptureSegmentAlgorithms.window: ["l1", "l2", "rbf"],
    }
