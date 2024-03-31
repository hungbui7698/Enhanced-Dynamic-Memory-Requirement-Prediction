import warnings
from functools import cache
from pathlib import Path

from .. import Globals, only_assert_if_debug, timed_function, MissingFileWarning
from . import (
    TaskTimeseriesCSVFileList,
    TaskTimeseriesCSVFiles,
    MemoryData,
    TaskTimeseriesCSV,
    MetadataData,
)


class TaskTimeseriesCSVDirectory:
    dir_path: Path
    wf_name: str
    task_name: str
    min_num_data_points: int = Globals.default_min_num_data_points

    def __init__(
        self,
        dir_path: Path,
        wf_name: str = None,
        task_name: str = None,
        min_num_data_points: int = Globals.default_min_num_data_points,
    ):
        only_assert_if_debug(dir_path.exists() and dir_path.is_dir())
        self.dir_path = dir_path
        if wf_name is not None:
            self.wf_name = wf_name
        else:
            self.wf_name = dir_path.parent.name
        if task_name is not None:
            self.task_name = task_name
        else:
            self.task_name = dir_path.name
        self.min_num_data_points = min_num_data_points

    @property
    @cache
    @timed_function("Parsing task timeseries CSV directory")
    def content(self) -> TaskTimeseriesCSVFileList:
        unique_names = sorted(
            {
                i.name.rsplit("_", 1)[0]
                for i in self.dir_path.iterdir()
                if i.is_file() and "memory" in i.name
            }
        )

        def has_all_4_csvs(name: str):
            for csv in (
                self.dir_path.joinpath(f"{name}_memory.csv"),
                self.dir_path.joinpath(f"{name}_cpu.csv"),
                self.dir_path.joinpath(f"{name}_file_event.csv"),
                self.dir_path.joinpath(f"{name}_metadata.csv"),
            ):
                if not (csv.exists() and csv.is_file()):
                    warnings.warn(f"{name} missing {csv.name} file", MissingFileWarning)
                    return False
            return True

        return TaskTimeseriesCSVFileList(
            list(
                filter(
                    lambda x: len(x.memory.values) >= self.min_num_data_points,
                    (
                        TaskTimeseriesCSVFiles(
                            MemoryData(
                                TaskTimeseriesCSV(
                                    self.dir_path.joinpath(f"{name}_memory.csv"),
                                    self.task_name,
                                    self.wf_name,
                                )
                            ),
                            TaskTimeseriesCSV(
                                self.dir_path.joinpath(f"{name}_cpu.csv"),
                                self.task_name,
                                self.wf_name,
                            ),
                            TaskTimeseriesCSV(
                                self.dir_path.joinpath(f"{name}_file_event.csv"),
                                self.task_name,
                                self.wf_name,
                            ),
                            MetadataData(
                                TaskTimeseriesCSV(
                                    self.dir_path.joinpath(f"{name}_metadata.csv"),
                                    self.task_name,
                                    self.wf_name,
                                )
                            ),
                        )
                        for name in unique_names
                        if has_all_4_csvs(name)
                    ),
                )
            ),
            task_name=self.task_name,
            wf_name=self.wf_name,
        )

    def __len__(self):
        return len(self.content)

    def __iter__(self):
        return iter(self.content)

    def __getitem__(self, item):
        return self.content[item]
