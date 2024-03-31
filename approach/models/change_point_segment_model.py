from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score

from .. import rc, Globals
from ..files import (
    TaskTimeseriesCSVFileList,
    TaskTimeseriesCSVFiles,
    RuptureSegmentAlgorithms,
    DenoiseParameters,
    RuptureParams,
)
from ..models import SegmentModel, LinearRegressionValueModel, OffsetDirection


class ChangePointSegmentModel(SegmentModel):
    name = "ChangePointSegmentModel"
    num_segments: int
    algorithm: RuptureSegmentAlgorithms
    rupture_params: RuptureParams

    def __init__(
        self,
        algorithm: RuptureSegmentAlgorithms,
        rupture_params: RuptureParams,
        use_preprocessed: bool | DenoiseParameters = False,
    ):
        self.num_segments = rupture_params.num_segments
        self.rupture_params = rupture_params
        self.model_time = []
        self.algorithm = algorithm
        self.use_preprocessed = use_preprocessed

    def train(self, files: TaskTimeseriesCSVFileList, *args, **kwargs):
        num_segments = self.num_segments
        if num_segments is None:
            num_segments = 4  # todo make this variable?

        input_sizes = files.input_total_sizes
        segment_lengths = np.asarray(
            [  # todo this looks horrible
                file.memory.cached.maybe_preprocessed(
                    self.use_preprocessed.train(num_segments, file)
                    if isinstance(self.use_preprocessed, DenoiseParameters)
                    else self.use_preprocessed
                ).call_segment_algorithm(self.algorithm, self.rupture_params)
                for file in files
            ]
        ).T
        self.model_time = [
            LinearRegressionValueModel(OffsetDirection.NOP).train(
                input_sizes, segment_len
            )
            for segment_len in segment_lengths
        ]
        if Globals.debug:
            for model, segment_len in zip(self.model_time, segment_lengths):
                rc.log(
                    score := r2_score(
                        segment_len,
                        [model.predict(input_size) for input_size in input_sizes],
                    )
                )
                if Globals.debug and score < 0.5:
                    g = sns.scatterplot(x=input_sizes, y=segment_len)
                    sns.lineplot(
                        x=input_sizes,
                        y=[model.predict(input_size) for input_size in input_sizes],
                    )
                    plt.show()

    def predict(
        self,
        runtime: int,
        input_size: float,
        oracle: TaskTimeseriesCSVFiles | None,
        *args,
        **kwargs,
    ) -> np.ndarray[int]:
        if oracle is not None:
            return oracle.memory.cached.maybe_preprocessed(
                self.use_preprocessed
            ).call_segment_algorithm(self.algorithm, self.rupture_params)
        return np.asarray(
            [model.predict(input_size) for model in self.model_time], dtype=int
        )

    @cache
    def __repr__(self):
        return f"{self.name}({self.num_segments}, {self.rupture_params}, {self.use_preprocessed})"
