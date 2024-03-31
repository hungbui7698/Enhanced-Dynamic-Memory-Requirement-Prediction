from typing import NamedTuple

from .. import only_assert_if_debug
from . import (
    MemoryData,
    TaskTimeseriesCSV,
    MetadataData,
    DenoiseParameters,
    StaticDenoiseParameters,
)


class TaskTimeseriesCSVFiles(NamedTuple):
    memory: MemoryData
    cpu: TaskTimeseriesCSV
    file_event: TaskTimeseriesCSV
    metadata: MetadataData

    @property
    def task_name(self):
        only_assert_if_debug(
            self.cpu.task_name
            == self.memory.csv_file.task_name
            == self.file_event.task_name
            == self.metadata.csv_file.task_name
        )
        return self.cpu.task_name

    @property
    def wf_name(self):
        only_assert_if_debug(
            self.cpu.wf_name
            == self.memory.csv_file.wf_name
            == self.file_event.wf_name
            == self.metadata.csv_file.wf_name
        )
        return self.cpu.wf_name

    def maybe_preprocessed(
        self, use_preprocessed: bool | StaticDenoiseParameters = False
    ):
        if (isinstance(use_preprocessed, bool) and use_preprocessed) or isinstance(
            use_preprocessed, DenoiseParameters
        ):
            if isinstance(use_preprocessed, DenoiseParameters):
                return self.memory.cached.denoised(use_preprocessed)
            else:
                return self.memory.cached.denoised()
        else:
            return self.memory

    def __len__(self):
        return len(self.memory)
