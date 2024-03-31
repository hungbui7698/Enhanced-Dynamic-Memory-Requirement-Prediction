from functools import partial
from pathlib import Path
from typing import List, Sequence, Dict, Tuple, Any

from dask.distributed import as_completed, Future
from distributed import get_client

from . import (
    benchmark_train_test_split,
    BenchmarkResult,
    shuffle_splits,
    WorkflowResult,
    TaskDirResult,
    benchmark_data,
)
from .. import FailedTrainingWarning, timed_function
from ..constants_and_global_misc import (
    only_assert_if_debug,
    Globals,
    rc,
    unzip,
)
from ..files import (
    DataDirectory,
    WFCSVDirectory,
    TaskTimeseriesCSVDirectory,
    TaskTimeseriesCSVFileList,
)
from ..models import (
    TaskModel,
    RetryModel,
    DefaultRetryModel,
    RetryMode,
)


def unpack_benchmark_train_test_split(args: Tuple):
    (
        models,
        train,
        test,
        r_mode,
        r_model,
    ) = args
    return benchmark_train_test_split(
        models=models, training=train, test=test, retry_mode=r_mode, retry_model=r_model
    )


@timed_function()
def benchmark_flat(
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
    if not Globals.do_parallel:
        return benchmark_data(
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
        client = get_client()
        results = []
        later: Dict[
            Future,
            Tuple[
                RetryMode,
                RetryModel,
                WFCSVDirectory,
                TaskTimeseriesCSVDirectory,
                float,
            ],
        ] = {}
        args = []
        params = []
        for r_mode in retry_mode:
            for r_model in retry_model:
                for wf in f_:
                    for task_dir in wf:
                        for p in percentages:
                            for train, test in shuffle_splits(
                                task_dir, p, shuffle_repeats
                            ):
                                args.append(
                                    (
                                        models,
                                        train,
                                        test,
                                        r_mode,
                                        r_model,
                                    )
                                )
                                params.append((r_mode, r_model, wf, task_dir, p))
                                # later[
                                #     client.submit(
                                #         benchmark_train_test_split,
                                #         models,
                                #         train,
                                #         test,
                                #         r_mode,
                                #         r_model,
                                #     )
                                # ] = (r_mode, r_model, wf, task_dir, p)
        args = client.scatter(args)
        later = {
            fut: param
            for fut, param in zip(
                client.map(unpack_benchmark_train_test_split, args), params
            )
        }
        for future, result in as_completed(
            later.keys(), with_results=True, raise_errors=False
        ):
            if future.status == "error":
                exception = future.exception()
                if fatal_errors or not isinstance(exception, FailedTrainingWarning):
                    for fut in later.keys():
                        try:
                            fut.cancel()
                        except:
                            continue
                    raise exception
                else:
                    continue
            (r_mode, r_model, wf, task_dir, p) = later.pop(future)
            results.append(
                BenchmarkResult(
                    [
                        WorkflowResult(
                            [
                                TaskDirResult(
                                    result,
                                    retry_mode=r_mode,
                                    retry_model=r_model,
                                    percentage=p,
                                    task_name=task_dir.task_name,
                                    wf_name=task_dir.wf_name,
                                )
                            ],
                            task_name=task_dir.task_name,
                            retry_mode=r_mode,
                            retry_model=r_model,
                            percentages=percentages,
                            wf_name=task_dir.wf_name,
                        )
                    ],
                    wf_name=wf.wf_name,
                    retry_mode=r_mode,
                    retry_model=r_model,
                    percentages=percentages,
                )
            )
        # for r_mode in retry_mode:
        #     for r_model in retry_model:
        #         for wf in f_:
        #             BenchmarkResult(
        #                 [
        #                     WorkflowResult(
        #                         [
        #                             TaskDirResult(
        #                                 benchmark_train_test_split(
        #                                     models,
        #                                     training=train,
        #                                     test=test,
        #                                     retry_mode=r_mode,
        #                                     retry_model=r_model,
        #                                 ),
        #                                 retry_mode=r_mode,
        #                                 retry_model=r_model,
        #                                 percentage=p,
        #                                 task_name=task_dir.task_name,
        #                                 wf_name=task_dir.wf_name,
        #                             )
        #                             for p in percentages
        #                             for train, test in shuffle_splits(
        #                                 task_dir, p, shuffle_repeats
        #                             )
        #                         ],
        #                         task_name=task_dir.task_name,
        #                         retry_mode=r_mode,
        #                         retry_model=r_model,
        #                         percentages=percentages,
        #                         wf_name=task_dir.wf_name,
        #                     )
        #                     for task_dir in wf
        #                 ],
        #                 wf_name=wf.wf_name,
        #                 retry_mode=r_mode,
        #                 retry_model=r_model,
        #                 percentages=percentages,
        #             )

        return results
