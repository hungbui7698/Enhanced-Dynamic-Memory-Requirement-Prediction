from functools import cache
from pathlib import Path
from typing import List

from .. import Globals, only_assert_if_debug, timed_function
from . import WFCSVDirectory, TaskTimeseriesCSVDirectory


class DataDirectory:
    dir_path: Path
    min_num_timeseries: int = Globals.default_min_num_timeseries
    min_num_data_points: int = Globals.default_min_num_data_points

    def __init__(
        self,
        dir_path: Path,
        min_num_timeseries: int = Globals.default_min_num_timeseries,
        min_num_data_points: int = Globals.default_min_num_data_points,
    ):
        only_assert_if_debug(dir_path.exists() and dir_path.is_dir())
        self.dir_path = dir_path
        self.min_num_timeseries = min_num_timeseries
        self.min_num_data_points = min_num_data_points

    @property
    @cache
    @timed_function("Parsing Data CSV directory")
    def content(self) -> List[WFCSVDirectory]:
        return list(
            filter(
                lambda x: len(x) > 0,
                (
                    WFCSVDirectory(
                        wf_dir,
                        min_num_timeseries=self.min_num_timeseries,
                        min_num_data_points=self.min_num_data_points,
                    )
                    for wf_dir in self.dir_path.iterdir()
                    if wf_dir.is_dir() and not wf_dir.name[0] == "."
                ),
            )
        )

    def flatten(self) -> List[TaskTimeseriesCSVDirectory]:
        return [task_dir for wf_dir in self.content for task_dir in wf_dir.content]

    def __iter__(self):
        return iter(self.content)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, item):
        return self.content[item]
