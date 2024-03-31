import warnings
from functools import cache
from typing import Sequence

import numpy as np

from .. import only_assert_if_debug, UnknownWfNameWarning, UnknownTaskNameWarning
from . import TaskTimeseriesCSVFiles, DenoiseParameters


class TaskTimeseriesCSVFileList:  # todo should probably convert these pseudo-list classes into actual subclasses of 'list'
    lst: Sequence[TaskTimeseriesCSVFiles]
    wf_name: str
    task_name: str

    def __init__(
        self,
        lst: Sequence[TaskTimeseriesCSVFiles],
        wf_name: str = None,
        task_name: str = None,
    ):
        self.lst = lst
        if wf_name is not None:
            self.wf_name = wf_name
        else:
            only_assert_if_debug(len(set(x.wf_name for x in self.lst)) == 1)
            if len(self.lst) > 0:
                self.wf_name = self.lst[0].wf_name
            else:
                warnings.warn(
                    f"Unknown wf_name in {self.__class__.__name__}",
                    UnknownWfNameWarning,
                )
        if task_name is not None:
            self.task_name = task_name
        else:
            only_assert_if_debug(len(set(x.task_name for x in self.lst)) == 1)
            if len(self.lst) > 0:
                self.task_name = self.lst[0].task_name
            else:
                warnings.warn(
                    f"Unknown task_name in {self.__class__.__name__}",
                    UnknownTaskNameWarning,
                )

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.lst[item]
        if isinstance(item, slice):
            return TaskTimeseriesCSVFileList(self.lst[item])
        if isinstance(item, Sequence):
            return TaskTimeseriesCSVFileList([self.lst[i] for i in item])
        raise TypeError("Unsupported item type")

    def __iter__(self):
        return iter(self.lst)

    def __contains__(self, item):
        return item in self.lst

    @property
    @cache
    def input_total_sizes(self) -> np.ndarray:
        return np.asarray([f.metadata.input_total_size for f in self.lst])

    @cache
    def lengths(self, use_preprocessed: bool | DenoiseParameters = False) -> np.ndarray:
        return np.asarray(
            [len(f.maybe_preprocessed(use_preprocessed).values) for f in self.lst]
        )

    @cache
    def peak_usages(
        self, use_preprocessed: bool | DenoiseParameters = False
    ) -> np.ndarray:
        return np.asarray(
            [
                f.memory.cached.maybe_preprocessed(use_preprocessed).max()
                for f in self.lst
            ]
        )
