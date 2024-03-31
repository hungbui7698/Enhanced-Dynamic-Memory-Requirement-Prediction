import multiprocessing
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Sequence

from distributed import get_client, secede, as_completed
from pynisher import pynisher

from . import (
    TaskDirResult,
    pool_benchmark_train_test_split,
    PoolBenchmarkTrainTestSplitArgs,
    benchmark_train_test_split,
)
from .. import timed_function, Globals, only_assert_if_debug, FailedTrainingWarning
from ..files import TaskTimeseriesCSVDirectory, TaskTimeseriesCSVFileList
from ..models import TaskModel, RetryMode, RetryModel, DefaultRetryModel
from ..progress import progress


def train_test_split(
    data: TaskTimeseriesCSVDirectory, percent_train: float, shuffle: bool = False
) -> Tuple[TaskTimeseriesCSVFileList, TaskTimeseriesCSVFileList]:
    i = int(len(data) * percent_train)
    if shuffle:
        idx = list(range(len(data)))
        random.shuffle(idx)
        return data[idx[:i]], data[idx[i:]]
    else:
        return data[:i], data[i:]


def shuffle_splits(
    data: TaskTimeseriesCSVDirectory, percent_train: float, shuffle_repeats: int = 0
):
    if shuffle_repeats > 0:
        splits = [
            train_test_split(data, percent_train, True) for _ in range(shuffle_repeats)
        ]
    else:
        splits = [train_test_split(data, percent_train, False)]
    return splits


@timed_function()  # enabled=not Globals.do_parallel)
def _benchmark_task_dir(
    models: List[TaskModel],
    files: TaskTimeseriesCSVDirectory = None,
    dir_path: Path = None,
    min_num_data_points: int = Globals.default_min_num_data_points,
    percentages: List[float] = None,
    retry_mode: RetryMode | Sequence[RetryMode] = None,
    retry_model: RetryModel | Sequence[RetryModel] = None,
    shuffle_repeats: int = 0,  # 0 == off by default
    own_pool: bool = True,
    fatal_errors: bool = True,
) -> List[TaskDirResult]:
    only_assert_if_debug(files is not None or dir_path is not None)
    if dir_path is not None:
        f_ = TaskTimeseriesCSVDirectory(
            dir_path, min_num_data_points=min_num_data_points
        )
    if files is not None:
        f_ = files
    if percentages is None:
        percentages = Globals.default_percentages
    if retry_model is None:
        retry_model = [DefaultRetryModel()]
    if not isinstance(retry_model, Sequence):
        retry_model = [retry_model]
    if retry_mode is None:
        retry_mode = [DefaultRetryModel()]
    if not isinstance(retry_mode, Sequence):
        retry_mode = [retry_mode]
    results = []
    if Globals.do_parallel:
        client = get_client()
        futures = client.map(
            pool_benchmark_train_test_split,
            [
                PoolBenchmarkTrainTestSplitArgs(
                    models,
                    training=train,
                    test=test,
                    retry_mode=r_mode,
                    retry_model=r_model,
                    percentage=p,
                )
                for r_mode in retry_mode
                for r_model in retry_model
                for p in percentages
                for train, test in shuffle_splits(f_, p, shuffle_repeats)
            ],
        )
        if not own_pool:
            secede()
        for future, result in as_completed(
            futures, with_results=True, raise_errors=False
        ):
            if future.status == "error":
                exception = future.exception()
                if fatal_errors or not isinstance(exception, FailedTrainingWarning):
                    for fut in futures:
                        try:
                            if fut.status != "error":
                                fut.cancel()
                        except:
                            pass
                    raise exception
                else:
                    continue
            (res_, args_) = result
            results.append(
                TaskDirResult(
                    res_,
                    retry_mode=args_.retry_mode,
                    retry_model=args_.retry_model,
                    percentage=args_.percentage,
                    task_name=args_.training.task_name,
                    wf_name=args_.training.wf_name,
                )
            )
            progress.advance()
    else:
        for r_mode in retry_mode:
            for r_model in retry_model:
                for p in percentages:
                    for train, test in shuffle_splits(f_, p, shuffle_repeats):
                        results.append(
                            TaskDirResult(
                                benchmark_train_test_split(
                                    models,
                                    training=train,
                                    test=test,
                                    retry_mode=r_mode,
                                    retry_model=r_model,
                                ),
                                retry_mode=r_mode,
                                retry_model=r_model,
                                percentage=p,
                                task_name=files.task_name,
                                wf_name=files.wf_name,
                            )
                        )
    return results


@timed_function()  # enabled=not Globals.do_parallel)
def benchmark_task_dir(
    models: List[TaskModel],
    files: TaskTimeseriesCSVDirectory = None,
    dir_path: Path = None,
    min_num_data_points: int = Globals.default_min_num_data_points,
    percentages: List[float] = None,
    retry_mode: RetryMode | Sequence[RetryMode] = None,
    retry_model: RetryModel | Sequence[RetryModel] = None,
    shuffle_repeats: int = 0,  # 0 == off by default
    own_pool: bool = True,
    fatal_errors: bool = True,
) -> List[TaskDirResult]:
    only_assert_if_debug(files is not None or dir_path is not None)
    if dir_path is not None:
        f_ = TaskTimeseriesCSVDirectory(
            dir_path, min_num_data_points=min_num_data_points
        )
    if files is not None:
        f_ = files
    if percentages is None:
        percentages = Globals.default_percentages
    if retry_model is None:
        retry_model = [DefaultRetryModel()]
    if not isinstance(retry_model, Sequence):
        retry_model = [retry_model]
    if retry_mode is None:
        retry_mode = [DefaultRetryModel()]
    if not isinstance(retry_mode, Sequence):
        retry_mode = [retry_mode]
    if Globals.benchmark_task_dir_wall_time_limit > 0 and Globals.use_pynisher:
        try:
            mem = get_worker().memory_manager.memory_limit
        except:
            mem = None
        multiprocessing.current_process()._config["daemon"] = False
        return pynisher.limit(
            _benchmark_task_dir,
            wall_time=(Globals.benchmark_task_dir_wall_time_limit, "s"),
            memory=mem,
        )(
            models,
            files,
            dir_path,
            min_num_data_points,
            percentages,
            retry_mode,
            retry_model,
            shuffle_repeats,
            own_pool,
            fatal_errors,
        )
    else:
        return _benchmark_task_dir(
            models,
            files,
            dir_path,
            min_num_data_points,
            percentages,
            retry_mode,
            retry_model,
            shuffle_repeats,
            own_pool,
            fatal_errors,
        )


@dataclass
class PoolBenchmarkTaskDirArgs:
    models: List[TaskModel]
    files: TaskTimeseriesCSVDirectory = None
    percentages: List[float] = None
    retry_mode: RetryMode | Sequence[RetryMode] = None
    retry_model: RetryModel | Sequence[RetryModel] = None
    shuffle_repeats: int = 0  # 0 == off by default


def pool_benchmark_task_dir(
    args: PoolBenchmarkTaskDirArgs,
) -> tuple[list[TaskDirResult], PoolBenchmarkTaskDirArgs]:
    return (
        benchmark_task_dir(
            models=args.models,
            files=args.files,
            percentages=args.percentages,
            retry_mode=args.retry_mode,
            retry_model=args.retry_model,
            shuffle_repeats=args.shuffle_repeats,
            own_pool=False,
        ),
        args,
    )
