from functools import cache
from pathlib import Path
from typing import List

from .. import Globals, only_assert_if_debug, timed_function
from . import TaskTimeseriesCSVDirectory


class WFCSVDirectory:
    dir_path: Path
    wf_name: str
    min_num_timeseries: int = Globals.default_min_num_timeseries
    min_num_data_points: int = Globals.default_min_num_data_points

    def __init__(
        self,
        dir_path: Path,
        wf_name: str = None,
        min_num_timeseries: int = Globals.default_min_num_timeseries,
        min_num_data_points: int = Globals.default_min_num_data_points,
    ):
        only_assert_if_debug(dir_path.exists() and dir_path.is_dir())
        self.dir_path = dir_path
        if wf_name is not None:
            self.wf_name = wf_name
        else:
            self.wf_name = dir_path.name
        self.min_num_timeseries = min_num_timeseries
        self.min_num_data_points = min_num_data_points

    @property
    @cache
    @timed_function("Parsing WF CSV directory")
    def content(self) -> List[TaskTimeseriesCSVDirectory]:
        return list(
            filter(
                lambda x: len(x) >= self.min_num_timeseries,
                (
                    TaskTimeseriesCSVDirectory(
                        p, self.wf_name, min_num_data_points=self.min_num_data_points
                    )
                    for p in self.dir_path.iterdir()
                    if p.is_dir()
                ),
            )
        )

    def __iter__(self):
        return iter(self.content)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, item):
        return self.content[item]
