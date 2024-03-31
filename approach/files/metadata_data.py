from functools import cache

from . import TaskTimeseriesCSV


class MetadataData:
    csv_file: TaskTimeseriesCSV

    def __init__(self, data: TaskTimeseriesCSV):
        self.csv_file = data

    @property
    @cache
    def input_total_size(self) -> float:
        return self.csv_file.csv.query('_field == "files_input_total_size"')[
            "_value"
        ].iloc[0]

    @property
    @cache
    def default_value(self) -> float:
        return (self.csv_file.csv.query('_field == "max_mem"')["_value"].iloc[0]) / (
            1024 * 1024
        )
