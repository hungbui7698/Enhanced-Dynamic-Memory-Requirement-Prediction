import warnings
from functools import cache
from typing import Dict, TypeAlias, Callable

import numpy as np
import ruptures as rpt

from .. import (
    only_assert_if_debug,
    BadBoundsWarning,
    cached_to_file,
    from_end_indices,
)
from . import (
    TaskTimeseriesCSV,
    DenoiseParameters,
    StaticDenoiseParameters,
    RuptureSegmentAlgorithms,
    RuptureParams,
    RuptureAlgorithmModelTypes,
)
from ..preprocessing import denoise_more


segment_algorithm_function: TypeAlias = Callable[[...], np.ndarray[int]]


class CachedDataTransformations:
    values: np.ndarray
    denoised: "CachedDataTransformations"
    csv_file: TaskTimeseriesCSV
    preprocessed: bool | DenoiseParameters

    def __init__(
        self,
        values: np.ndarray,
        csv_file: TaskTimeseriesCSV,
        preprocessed: bool | DenoiseParameters = False,
    ):
        self.values = values
        self.segment_algorithms: Dict[
            RuptureSegmentAlgorithms, segment_algorithm_function
        ] = {
            RuptureSegmentAlgorithms.binseg: self.binseg_algorithm,
            RuptureSegmentAlgorithms.bottom_up: self.bottom_up_algorithm,
            RuptureSegmentAlgorithms.dynp: self.dynp_algorithm,
            RuptureSegmentAlgorithms.kernel_cpd: self.kernel_cpd_algorithm,
            RuptureSegmentAlgorithms.pelt: self.pelt_algorithm,
            RuptureSegmentAlgorithms.window: self.window_algorithm,
        }
        self.csv_file = csv_file
        self.preprocessed = preprocessed

    def maybe_preprocessed(
        self, use_preprocessed: bool | StaticDenoiseParameters = False
    ):
        if (isinstance(use_preprocessed, bool) and use_preprocessed) or isinstance(
            use_preprocessed, DenoiseParameters
        ):
            if isinstance(use_preprocessed, DenoiseParameters):
                return self.denoised(use_preprocessed)
            else:
                return self.denoised()
        else:
            return self

    @cache
    def max(self, start: int = 0, end: int = None):
        if end is None:
            end = len(self.values)
        only_assert_if_debug(start <= end)
        if start <= 0 and end >= len(self.values):
            return self.values.max()
        s_ = min(len(self.values) - 1, start)
        if s_ != start:
            warnings.warn(
                "RememberMaxes: start is past end of self.data", BadBoundsWarning
            )
        return self.values[s_:end].max()

    def segment_algorithm(
        self, algorithm: RuptureSegmentAlgorithms
    ) -> segment_algorithm_function:
        return self.segment_algorithms[algorithm]

    def call_segment_algorithm(
        self, algorithm: RuptureSegmentAlgorithms, params: RuptureParams
    ) -> np.ndarray[int]:
        return self.segment_algorithm(algorithm)(**params.__dict__)

    # todo: DRY
    @cache
    def bottom_up_algorithm(
        self,
        num_segments: int,
        start: int = 0,
        end: int = None,
        model: str = None,
        min_size: int = 2,
        jump: int = 5,
        pen: float = None,
        epsilon: float = None,
        **kwargs,
    ) -> np.ndarray[int]:
        if model is None:
            model = "rbf"
        else:
            only_assert_if_debug(
                model
                in RuptureAlgorithmModelTypes.by_algorithm[
                    RuptureSegmentAlgorithms.bottom_up
                ]
            )

        @cached_to_file(
            f"{self.csv_file.path.parent.parent.name}_{self.csv_file.path.stem}",
            self.preprocessed,
            f_name="bottom_up_algorithm",
        )
        def _run(
            _num_segments: int,
            _start: int = 0,
            _end: int = None,
            _model: str = "rbf",
            _min_size: int = 2,
            _jump: int = 5,
            _pen: float = None,
            _epsilon: float = None,
        ) -> np.ndarray[int]:
            if _end is None:
                _end = len(self.values)
            algo = rpt.BottomUp(model=_model, min_size=_min_size, jump=_jump).fit(
                self.values[_start:_end]
            )
            result = algo.predict(n_bkps=_num_segments - 1, pen=_pen, epsilon=_epsilon)
            return from_end_indices(result)

        return _run(num_segments, start, end, model, min_size, jump, pen, epsilon)

    @cache
    def binseg_algorithm(
        self,
        num_segments: int,
        start: int = 0,
        end: int = None,
        model: str = None,
        min_size: int = 2,
        jump: int = 5,
        pen: float = None,
        epsilon: float = None,
        **kwargs,
    ) -> np.ndarray[int]:
        if model is None:
            model = "rbf"
        else:
            only_assert_if_debug(
                model
                in RuptureAlgorithmModelTypes.by_algorithm[
                    RuptureSegmentAlgorithms.binseg
                ]
            )

        @cached_to_file(
            f"{self.csv_file.path.parent.parent.name}_{self.csv_file.path.stem}",
            self.preprocessed,
            f_name="binseg_algorithm",
        )
        def _run(
            _num_segments: int,
            _start: int = 0,
            _end: int = None,
            _model: str = "rbf",
            _min_size: int = 2,
            _jump: int = 5,
            _pen: float = None,
            _epsilon: float = None,
        ) -> np.ndarray[int]:
            if _end is None:
                _end = len(self.values)
            algo = rpt.Binseg(model=_model, min_size=_min_size, jump=_jump).fit(
                self.values[_start:_end]
            )
            result = algo.predict(n_bkps=_num_segments - 1, pen=_pen, epsilon=_epsilon)
            return from_end_indices(result)

        return _run(num_segments, start, end, model, min_size, jump, pen, epsilon)

    @cache
    def dynp_algorithm(
        self,
        num_segments: int,
        start: int = 0,
        end: int = None,
        model: str = None,
        min_size: int = 2,
        jump: int = 5,
        **kwargs,
    ) -> np.ndarray[int]:
        if model is None:
            model = "rbf"
        else:
            only_assert_if_debug(
                model
                in RuptureAlgorithmModelTypes.by_algorithm[
                    RuptureSegmentAlgorithms.dynp
                ]
            )

        @cached_to_file(
            f"{self.csv_file.path.parent.parent.name}_{self.csv_file.path.stem}",
            self.preprocessed,
            f_name="dynp_algorithm",
        )
        def _run(
            _num_segments: int,
            _start: int = 0,
            _end: int = None,
            _model: str = "rbf",
            _min_size: int = 2,
            _jump: int = 5,
        ) -> np.ndarray[int]:
            if _end is None:
                _end = len(self.values)
            algo = rpt.Dynp(model=_model, min_size=_min_size, jump=_jump).fit(
                self.values[_start:_end]
            )
            result = algo.predict(n_bkps=_num_segments - 1)
            return from_end_indices(result)

        return _run(num_segments, start, end, model, min_size, jump)

    @cache
    def kernel_cpd_algorithm(
        self,
        num_segments: int,
        start: int = 0,
        end: int = None,
        model: str = None,
        min_size: int = 2,
        jump: int = 1,
        pen: float = None,
        **kwargs,
    ) -> np.ndarray[int]:
        if model is None:
            model = "linear"
        else:
            only_assert_if_debug(
                model
                in RuptureAlgorithmModelTypes.by_algorithm[
                    RuptureSegmentAlgorithms.kernel_cpd
                ]
            )

        @cached_to_file(
            f"{self.csv_file.path.parent.parent.name}_{self.csv_file.path.stem}",
            self.preprocessed,
            f_name="kernel_cpd_algorithm",
        )
        def _run(
            _num_segments: int,
            _start: int = 0,
            _end: int = None,
            _model: str = "linear",
            _min_size: int = 2,
            _jump: int = 1,
            _pen: float = None,
        ) -> np.ndarray[int]:
            if _end is None:
                _end = len(self.values)
            algo = rpt.KernelCPD(kernel=_model, min_size=_min_size, jump=_jump).fit(
                self.values[_start:_end]
            )
            result = algo.predict(n_bkps=_num_segments - 1, pen=_pen)
            return from_end_indices(result)

        return _run(num_segments, start, end, model, min_size, jump, pen)

    @cache
    def pelt_algorithm(
        self,
        pen: float,
        start: int = 0,
        end: int = None,
        model: str = None,
        min_size: int = 2,
        jump: int = 5,
        **kwargs,
    ) -> np.ndarray[int]:
        if model is None:
            model = "rbf"
        else:
            only_assert_if_debug(
                model
                in RuptureAlgorithmModelTypes.by_algorithm[
                    RuptureSegmentAlgorithms.pelt
                ]
            )

        @cached_to_file(
            f"{self.csv_file.path.parent.parent.name}_{self.csv_file.path.stem}",
            self.preprocessed,
            f_name="pelt_algorithm",
        )
        def _run(
            _pen: float,
            _start: int = 0,
            _end: int = None,
            _model: str = "rbf",
            _min_size: int = 2,
            _jump: int = 5,
        ) -> np.ndarray[int]:
            if _end is None:
                _end = len(self.values)
            algo = rpt.Pelt(model=_model, min_size=_min_size, jump=_jump).fit(
                self.values[_start:_end]
            )
            result = algo.predict(pen=_pen)
            return from_end_indices(result)

        return _run(pen, start, end, model, min_size, jump)

    @cache
    def window_algorithm(
        self,
        num_segments: int,
        start: int = 0,
        end: int = None,
        model: str = None,
        min_size: int = 2,
        jump: int = 5,
        width: int = 100,
        pen: float = None,
        epsilon: float = None,
        **kwargs,
    ) -> np.ndarray[int]:
        if model is None:
            model = "rbf"
        else:
            only_assert_if_debug(
                model
                in RuptureAlgorithmModelTypes.by_algorithm[
                    RuptureSegmentAlgorithms.window
                ]
            )

        @cached_to_file(
            f"{self.csv_file.path.parent.parent.name}_{self.csv_file.path.stem}",
            self.preprocessed,
            f_name="window_algorithm",
        )
        def _run(
            _num_segments: int,
            _start: int = 0,
            _end: int = None,
            _model: str = "rbf",
            _min_size: int = 2,
            _jump: int = 5,
            _width: int = 100,
            _pen: float = None,
            _epsilon: float = None,
        ) -> np.ndarray[int]:
            if _end is None:
                _end = len(self.values)
            model = "l2"  # can be "l2", "rbf", "linear", etc.W
            algo = rpt.Window(
                model=_model, min_size=_min_size, jump=_jump, width=_width
            ).fit(self.values[_start:_end])
            result = algo.predict(n_bkps=_num_segments - 1, pen=_pen, epsilon=_epsilon)
            return from_end_indices(result)

        return _run(
            num_segments, start, end, model, min_size, jump, width, pen, epsilon
        )

    @cache
    def denoised(
        self, params: StaticDenoiseParameters = None
    ) -> "CachedDataTransformations":
        if params is None:
            params = StaticDenoiseParameters()
        if (
            isinstance(self.preprocessed, bool) and not self.preprocessed
        ) or isinstance(self.preprocessed, DenoiseParameters):
            return CachedDataTransformations(
                denoise_more(
                    self.values,
                    params.stride,
                    params.widths,
                    params.depth,
                    params.freeze_ends,
                    params.maxer_type,
                ),
                csv_file=self.csv_file,
                preprocessed=params,
            )
        else:
            return self

    @cache
    def offset(
        self, percent_offset: float = 0.0, abs_offset: float = 0.0
    ) -> "CachedDataTransformations":
        if percent_offset == 0 and abs_offset == 0:
            return self
        return CachedDataTransformations(
            self.values * (1 + percent_offset) + abs_offset,
            csv_file=self.csv_file,
            preprocessed=self.preprocessed,
        )
