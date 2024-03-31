from functools import cache

import numpy as np

from . import TaskTimeseriesCSV, CachedDataTransformations


class MemoryData:
    csv_file: TaskTimeseriesCSV
    cached: CachedDataTransformations

    def __init__(self, data: TaskTimeseriesCSV):
        self.csv_file = data
        self.cached = CachedDataTransformations(
            self.values, csv_file=self.csv_file, preprocessed=False
        )

    @property
    @cache
    def values(
        self,
    ) -> np.ndarray[float]:
        return self.csv_file.csv["_value"].values

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, item):
        return self.values[item]
