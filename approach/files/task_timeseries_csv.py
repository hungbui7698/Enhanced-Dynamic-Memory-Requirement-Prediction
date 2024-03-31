from pathlib import Path

import pandas as pds

from .. import only_assert_if_debug


class TaskTimeseriesCSV:
    task_name: str
    wf_name: str
    path: Path

    def __init__(self, path: Path, task_name: str = None, wf: str = None):
        only_assert_if_debug(path.exists(), f"{path.as_posix()} doesn't exist")
        only_assert_if_debug(path.is_file(), f"{path.as_posix()} is not a file")
        self.path = path
        if task_name is not None:
            self.task_name = task_name
        else:
            self.task_name = path.name.split(" ", 1)[0]
        if wf is not None:
            self.wf_name = wf
        else:
            self.wf_name = path.parent.parent.name

    @property
    # @cache
    def csv(self) -> pds.DataFrame:
        return pds.read_csv(self.path.resolve().absolute().as_posix(), skiprows=3)
