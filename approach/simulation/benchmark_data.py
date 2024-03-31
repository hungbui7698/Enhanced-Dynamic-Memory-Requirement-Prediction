import multiprocessing
from pathlib import Path
from typing import List, Sequence

import pynisher
from dask.distributed import as_completed, Future
from distributed import get_client, get_worker

from . import (
    benchmark_workflow,
    PoolBenchmarkWorkflowArgs,
    pool_benchmark_workflow_dir,
    BenchmarkResult,
)
from .. import (
    FailedTrainingWarning,
)
from ..constants_and_global_misc import (
    only_assert_if_debug,
    Globals,
    timed_function,
)
from ..files import (
    DataDirectory,
)
from ..models import (
    TaskModel,
    RetryModel,
    DefaultRetryModel,
    RetryMode,
)


@timed_function()
def _benchmark_data(
    models: List[TaskModel],
    data_path: Path = None,
    files: DataDirectory = None,
    percentages: List[float] = None,
    min_num_timeseries: int = Globals.default_min_num_timeseries,
    min_num_data_points: int = Globals.default_min_num_data_points,
    retry_mode: RetryMode | Sequence[RetryMode] = None,
    retry_model: RetryModel | Sequence[RetryModel] = None,
    shuffle_repeats: int = 0,
    fatal_errors: bool = True,
) -> List[BenchmarkResult]:
    only_assert_if_debug(data_path is not None or files is not None)
    if data_path is not None:
        # doesn't immediately read from disk, so this is fine
        f_ = DataDirectory(
            data_path,
            min_num_timeseries=min_num_timeseries,
            min_num_data_points=min_num_data_points,
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
        futures: List[Future] = client.map(
            pool_benchmark_workflow_dir,
            [
                PoolBenchmarkWorkflowArgs(
                    models,
                    files=wf,
                    percentages=percentages,
                    retry_mode=r_mode,
                    retry_model=r_model,
                    shuffle_repeats=shuffle_repeats,
                )
                for r_mode in retry_mode
                for r_model in retry_model
                for wf in f_
            ],
        )
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
                BenchmarkResult(
                    res_,
                    retry_mode=args_.retry_mode,
                    retry_model=args_.retry_model,
                    percentages=args_.percentages,
                    wf_name=args_.files.wf_name,
                )
            )
    else:
        for r_mode in retry_mode:
            for r_model in retry_model:
                for wf in f_:
                    results.append(
                        BenchmarkResult(
                            benchmark_workflow(
                                models,
                                files=wf,
                                percentages=percentages,
                                retry_mode=r_mode,
                                retry_model=r_model,
                                shuffle_repeats=shuffle_repeats,
                            ),
                            wf_name=wf.wf_name,
                            retry_mode=r_mode,
                            retry_model=r_model,
                            percentages=percentages,
                        )
                    )

    return results


@timed_function()
def benchmark_data(
    models: List[TaskModel],
    data_path: Path = None,
    files: DataDirectory = None,
    percentages: List[float] = None,
    min_num_timeseries: int = Globals.default_min_num_timeseries,
    min_num_data_points: int = Globals.default_min_num_data_points,
    retry_mode: RetryMode | Sequence[RetryMode] = None,
    retry_model: RetryModel | Sequence[RetryModel] = None,
    shuffle_repeats: int = 0,
    fatal_errors: bool = True,
) -> List[BenchmarkResult]:
    only_assert_if_debug(data_path is not None or files is not None)
    if data_path is not None:
        # doesn't immediately read from disk, so this is fine
        f_ = DataDirectory(
            data_path,
            min_num_timeseries=min_num_timeseries,
            min_num_data_points=min_num_data_points,
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
    if Globals.benchmark_data_wall_time_limit > 0 and Globals.use_pynisher:
        try:
            mem = get_worker().memory_manager.memory_limit
        except:
            mem = None
        multiprocessing.current_process()._config["daemon"] = False
        return pynisher.limit(
            _benchmark_data,
            wall_time=(Globals.benchmark_data_wall_time_limit, "s"),
            memory=mem,
        )(
            models,
            data_path,
            files,
            percentages,
            min_num_timeseries,
            min_num_data_points,
            retry_mode,
            retry_model,
            shuffle_repeats,
            fatal_errors,
        )
    else:
        return _benchmark_data(
            models,
            data_path,
            files,
            percentages,
            min_num_timeseries,
            min_num_data_points,
            retry_mode,
            retry_model,
            shuffle_repeats,
            fatal_errors,
        )
