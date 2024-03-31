from functools import cache

import numpy as np

from . import SegmentModel
from ..files import TaskTimeseriesCSVFiles, DenoiseParameters


class KEqualSegmentModel(SegmentModel):
    name = "KEqualSegmentModel"
    # SegmentModel that produces k equal sized segments
    num_segments: int

    def __init__(self, k: int, use_preprocessed: bool | DenoiseParameters = False):
        self.num_segments = k
        self.use_preprocessed = use_preprocessed

    def _pred_v1(self, runtime: int) -> np.ndarray:
        r = self.num_segments
        o = []
        while r > 0:
            i = int(runtime / r)
            o.append(i)
            runtime -= i
            r -= 1
        return np.asarray(o)

    def _pred_v2(self, runtime: int) -> np.ndarray:
        i = int(runtime / self.num_segments)
        r = runtime - self.num_segments * i
        o = [i] * self.num_segments
        c = -1
        while r > 0:
            o[c] += 1
            r -= 1
            c -= 1
        return np.asarray(o)

    def predict(
        self,
        runtime: int,
        input_size: float,
        oracle: TaskTimeseriesCSVFiles | None,
        *args,
        **kwargs,
    ) -> np.ndarray[int]:
        # todo not sure which implementation to go with
        if oracle is not None:
            return self._pred_v1(
                len(oracle.maybe_preprocessed(self.use_preprocessed).values)
            )
        return self._pred_v1(runtime)

    @cache
    def __repr__(self):
        return f"{self.name}({self.num_segments}, {self.use_preprocessed})"
