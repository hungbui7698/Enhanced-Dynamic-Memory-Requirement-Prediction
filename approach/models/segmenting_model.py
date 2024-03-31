from functools import partial, cache
from typing import Type, List

import numpy as np
from overrides import overrides

from . import TaskModel, TimeModel, ValueModel, SegmentModel
from .. import transpose_uneven, unzip
from ..constants_and_global_misc import to_edge_pairs
from ..files import TaskTimeseriesCSVFileList, DenoiseParameters


# general form of all k-segment models
class SegmentingModel(TaskModel):
    name = "SegmentingModel"
    # models the usage of a task, using a time, segment, and multiple value models
    time_model: TimeModel
    value_model_type: Type[
        ValueModel
    ]  # the type of model that will be instantiated for each segment
    segment_value_models: List[ValueModel]  # list of value model instances
    segment_model: SegmentModel

    def __init__(
        self,
        time_model: TimeModel,
        value_model_type: Type[ValueModel] | partial[Type[ValueModel]],
        segment_model: SegmentModel,
        max_percent_offset: float = 0.0,
        max_abs_offset: float = 0.0,
        use_preprocessed: bool | DenoiseParameters = False,
        *args,
        **kwargs,
    ):
        self.time_model = time_model
        self.value_model_type = value_model_type
        self.segment_model = segment_model
        self.segment_value_models = []
        self.max_percent_offset = max_percent_offset
        self.max_abs_offset = max_abs_offset
        self.use_preprocessed = use_preprocessed

    def train(
        self, files: TaskTimeseriesCSVFileList, *args, **kwargs
    ) -> "SegmentingModel":
        self.time_model.train(files, *args, **kwargs)
        self.segment_model.train(files, *args, **kwargs)

        self.segment_value_models = []
        preprocessing = self.use_preprocessed
        num_segments = self.segment_model.num_segments
        if num_segments is None:
            num_segments = 4  # todo make this variable?
        segment_maxes = []
        for file in files:
            if isinstance(preprocessing, DenoiseParameters):
                preprocessing = preprocessing.train(num_segments, file)
            runtime = len(file.maybe_preprocessed(preprocessing).values)
            run_lengths = self.segment_model.predict(
                runtime, file.metadata.input_total_size, file
            )
            segment_maxes.append(
                [
                    (
                        file.metadata.input_total_size,
                        file.memory.cached.maybe_preprocessed(preprocessing)
                        .offset(self.max_percent_offset, self.max_abs_offset)
                        .max(
                            p.start,
                            p.end,
                        ),
                    )  # todo maybe other aggregations might be interesting
                    for p in to_edge_pairs(run_lengths)
                ]
            )
        segment_maxes = transpose_uneven(segment_maxes)
        for segment in segment_maxes:
            self.segment_value_models.append(
                self.value_model_type().train(*unzip(segment))
            )
        return self

    @overrides(check_signature=False)
    def predict(self, input_size: float, *args, **kwargs) -> np.ndarray[float]:
        runtime = self.time_model.predict(input_size)
        run_lengths = self.segment_model.predict(runtime, input_size, None)
        num_segments = len(run_lengths)
        vps = [
            vm.predict(input_size) for vm in self.segment_value_models[:num_segments]
        ]
        return np.asarray([x for v, rl in zip(vps, run_lengths) for x in [v] * rl])

    @cache
    def __repr__(self):
        return f"{self.name}({self.time_model!r}, {self.value_model_type()!r}, {self.segment_model!r}, {self.use_preprocessed})"
