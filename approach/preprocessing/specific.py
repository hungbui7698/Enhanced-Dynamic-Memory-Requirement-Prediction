import re
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pds
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from . import basic, operations
from .. import only_assert_if_debug


def load_data(
    wf_name: str = None,
    task_name: str = None,
    base_data_path: Path = None,
    csvs: List[pds.DataFrame] = None,
    make_blocky: bool = False,
    stride: int = 15,
    make_smooth: bool = False,
    smoothing: int = 15,
    x_col_name: str = "passed",
    y_col_name: str = "_value",
    origin_col_name: str = "origin",
    min_data_points: int = 10,
) -> pds.DataFrame | None:
    """
    Loads all CSVs corresponding to given task in given workflow (the data directory will have to be populated with the long2mao1 data -- see init_data.txt in base directory).
    Converts the measurement timestamps into amounts of time passed since first measurement and adds names of the csv files as origins to differentiate different measurement
    series.

    Default column names (can be altered using *_col_name parameters):

    -- X data: "passed"

    -- Y data: "_value"

    -- origin file name: "origin"
    :param wf_name: name of the workflow
    :param task_name: name of the task
    :param base_data_path: Path to the root directory of stored data
    :param csvs: list of csvs each containing a data series of the same task
    :param make_blocky: whether to apply blockyfication operation
    :param stride: stride of potential blockifycation operation
    :param make_smooth: whether to apply smoothing operation
    :param smoothing: width of potential smoothing operation
    :param x_col_name:
    :param y_col_name:
    :param origin_col_name:
    :param min_data_points:
    :return: A dataframe containing the X and Y data and the origins of each measurement series
    """
    only_assert_if_debug(
        (csvs is not None) or (wf_name is not None and task_name is not None)
    )
    # using https://github.com/long2mao1/k-segments-traces/
    data = []
    if csvs is not None:
        for i, csv in enumerate(csvs):
            csv: pds.DataFrame
            csv = csv[["_time", "_value"]]
            csv = csv.assign(
                _time=list(
                    map(lambda x: datetime.fromisoformat(x), csv["_time"].values)
                )
            )
            csv = csv.set_index(keys=["_time"], drop=True)
            data.append(csv.assign(**dict([(origin_col_name, i)])))
    else:
        if base_data_path is None:
            base_data_path = Path("../data")
        task_data_path = base_data_path.joinpath(f"{wf_name}/{task_name}")
        only_assert_if_debug(task_data_path.exists() and task_data_path.is_dir())
        for file in task_data_path.iterdir():
            if re.match(".*memory\.csv", file.as_posix()) is not None:
                d = pds.read_csv(
                    file.as_posix(),
                    header=3,
                    index_col="_time",
                    usecols=["_time", "_value"],
                    dtype={"_value": np.float64},
                    converters={"_time": lambda x: datetime.fromisoformat(x)},
                )
                if len(d) >= min_data_points:
                    data.append(d.assign(**dict([(origin_col_name, file.as_posix())])))
    # data = [pds.read_csv(file.as_posix(), header=3, index_col="_time", usecols=["_time", "_value"],
    #                      dtype={"_value": np.float64},
    #                      converters={"_time": lambda x: datetime.datetime.fromisoformat(x)}) for file in
    #         task_data_path.iterdir() if re.match(".*memory\.csv", file.as_posix()) is not None]
    if len(data) < 1:
        return None
    for i, d in enumerate(data):
        d[x_col_name] = [x.total_seconds() for x in d.index - min(d.index)]
        data[i] = d.reset_index(drop=True)
    max_res = max(len(d) for d in data)
    new_x = basic.Interval(
        max(d[x_col_name].min() for d in data),
        min(d[x_col_name].max() for d in data),
        max_res - 2,
    ).items  # produces |max_res| evenly spaced points (including 0 and 1)
    for i, d in enumerate(data):
        xscale = d["passed"].max()
        yscale = d["_value"].max()
        interpol = operations.Interpoler(d[x_col_name].values, d["_value"].values)
        x_data = MinMaxScaler().fit_transform(new_x.reshape(-1, 1)).ravel()
        y_data = (
            MinMaxScaler()
            .fit_transform(np.array(interpol[new_x]).reshape(-1, 1))
            .ravel()
        )
        if make_blocky:
            y_data = operations.blockify(y_data, stride)
        if make_smooth:
            y_data = operations.smooth(y_data, smoothing)
        new_d = pds.DataFrame(
            data={
                x_col_name: pds.Series(data=x_data, name=x_col_name),
                y_col_name: pds.Series(data=y_data, name=y_col_name),
            }
        ).assign(xscale=xscale, yscale=yscale)
        # d["_value"] = pds.Series(data=reshape_y(d["_value"].values).ravel(), index=d.index, name="_value")
        data[i] = new_d.assign(
            **dict([(origin_col_name, d[origin_col_name].values[0])])
        )
    return pds.concat(data, axis="rows", ignore_index=True)


def df_maxes(
    df: pds.DataFrame, y_col_name="_value", origin_col_name="origin"
) -> np.ndarray:
    return basic.many_max(
        np.array([d[y_col_name].values for _, d in df.groupby([origin_col_name])])
    )


def df_percentile(
    df: pds.DataFrame, percentile=80, y_col_name="_value", origin_col_name="origin"
) -> np.ndarray:
    return np.percentile(
        np.array([d[y_col_name].values for _, d in df.groupby([origin_col_name])]),
        percentile,
        axis=0,
    )
