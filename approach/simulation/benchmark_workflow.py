from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from distributed import get_client, Future, secede, as_completed

from . import (
    PoolBenchmarkTaskDirArgs,
    WorkflowResult,
    pool_benchmark_task_dir,
    benchmark_task_dir,
)
from .. import timed_function, Globals, only_assert_if_debug, FailedTrainingWarning
from ..files import WFCSVDirectory, TaskTimeseriesCSVDirectory
from ..models import TaskModel, RetryMode, RetryModel, DefaultRetryModel


@timed_function()
def benchmark_workflow(
    models: List[TaskModel],
    wf_path: Path = None,
    files: WFCSVDirectory | List[TaskTimeseriesCSVDirectory] = None,
    percentages: List[float] = None,
    min_num_timeseries: int = Globals.default_min_num_timeseries,
    min_num_data_points: int = Globals.default_min_num_data_points,
    retry_mode: RetryMode | Sequence[RetryMode] = None,
    retry_model: RetryModel | Sequence[RetryModel] = None,
    shuffle_repeats: int = 0,
    own_pool: bool = True,
    fatal_errors: bool = True,
) -> list[PoolBenchmarkTaskDirArgs] | list[WorkflowResult]:
    only_assert_if_debug(wf_path is not None or files is not None)
    if wf_path is not None:
        f_ = WFCSVDirectory(
            wf_path,
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
            pool_benchmark_task_dir,
            [
                PoolBenchmarkTaskDirArgs(
                    models,
                    files=task_dir,
                    percentages=percentages,
                    retry_mode=r_mode,
                    retry_model=r_model,
                    shuffle_repeats=shuffle_repeats,
                )
                for r_mode in retry_mode
                for r_model in retry_model
                for task_dir in f_
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
                WorkflowResult(
                    res_,
                    task_name=args_.files.task_name,
                    retry_mode=args_.retry_mode,
                    retry_model=args_.retry_model,
                    percentages=args_.percentages,
                    wf_name=args_.files.wf_name,
                )
            )
    else:
        for r_mode in retry_mode:
            for r_model in retry_model:
                for task_dir in f_:
                    results.append(
                        WorkflowResult(
                            benchmark_task_dir(
                                models,
                                files=task_dir,
                                percentages=percentages,
                                retry_mode=r_mode,
                                retry_model=r_model,
                                shuffle_repeats=shuffle_repeats,
                            ),
                            task_name=task_dir.task_name,
                            retry_mode=r_mode,
                            retry_model=r_model,
                            percentages=percentages,
                            wf_name=task_dir.wf_name,
                        )
                    )

    return results


@dataclass
class PoolBenchmarkWorkflowArgs:
    models: List[TaskModel]
    files: WFCSVDirectory | List[TaskTimeseriesCSVDirectory] = None
    percentages: List[float] = None
    retry_mode: RetryMode | Sequence[RetryMode] = None
    retry_model: RetryModel | Sequence[RetryModel] = None
    shuffle_repeats: int = 0


def pool_benchmark_workflow_dir(
    args: PoolBenchmarkWorkflowArgs,
) -> tuple[list[WorkflowResult], PoolBenchmarkWorkflowArgs]:
    return (
        benchmark_workflow(
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
