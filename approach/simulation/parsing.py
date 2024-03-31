import enum
from enum import Enum
from typing import Sequence, List

import pandas as pds

from . import WastageTypes, TaskDirResult, WorkflowResult, BenchmarkResult
from .. import timed_function, shorten_name

col_name_by_wastage_type = {
    WastageTypes.Total: "Wastage",
    WastageTypes.Over: "Wastage_over",
    WastageTypes.Under: "Wastage_under",
}


class PotentialTypes(Enum):
    Full = enum.auto()
    With_retries = enum.auto()


col_name_by_potential_type = {
    PotentialTypes.With_retries: "Potential_with_retries",
    PotentialTypes.Full: "Potential_full",
}
col_names = [
    "cat",
    "perc",
    "val",
    "orig",
    "short_name",
    "task",
    "retry_mode",
    "retry_model",
    "wf_name",
]


@timed_function()
def task_dir_result_list_to_df(task_dir_results: List[TaskDirResult]) -> pds.DataFrame:
    return pds.concat(
        [task_dir_result_to_df(res) for res in task_dir_results],
        ignore_index=True,
    )


@timed_function()
def task_dir_result_to_df(task_dir_result: TaskDirResult) -> pds.DataFrame:
    return pds.DataFrame(
        data=[
            [
                cat,
                task_dir_result.percentage,
                val,
                str(res.model),
                shorten_name(str(res.model)),
                task_dir_result.task_name,
                task_dir_result.retry_mode.name,
                str(task_dir_result.retry_model),
                task_dir_result.wf_name,
            ]
            for res in task_dir_result.split_results
            for val, cat in zip(
                (
                    res.avg_waste.total,
                    res.avg_waste.over,
                    res.avg_waste.under,
                    res.avg_retries,
                    res.avg_runtime,
                    res.avg_efficacy,
                    res.avg_efficacy * res.avg_waste.over + res.avg_waste.under,
                    res.avg_efficacy * res.avg_waste.over,
                ),
                (
                    *list(col_name_by_wastage_type.values()),
                    "Retries",
                    "Runtime",
                    "Efficacy",
                    *list(col_name_by_potential_type.values()),
                ),
            )
        ],
        columns=col_names,
    )


@timed_function()
def workflow_result_to_df(
    workflow_result: WorkflowResult | Sequence[WorkflowResult],
) -> pds.DataFrame:
    def do_one(a: WorkflowResult) -> pds.DataFrame:
        return pds.concat(
            [task_dir_result_to_df(res) for res in a.task_dir_results],
            ignore_index=True,
        )

    if isinstance(workflow_result, WorkflowResult):
        return do_one(workflow_result)
    elif isinstance(workflow_result, Sequence):
        return pds.concat([do_one(x) for x in workflow_result], ignore_index=True)
    else:
        raise TypeError("Wrong type for workflow_result")


@timed_function()
def benchmark_result_to_df(
    benchmark_result: BenchmarkResult | Sequence[BenchmarkResult],
) -> pds.DataFrame:
    def do_one(a: BenchmarkResult) -> pds.DataFrame:
        return pds.concat(
            [workflow_result_to_df(res) for res in a.workflow_results],
            ignore_index=True,
        )

    if isinstance(benchmark_result, BenchmarkResult):
        return do_one(benchmark_result)
    elif isinstance(benchmark_result, Sequence):
        return pds.concat([do_one(x) for x in benchmark_result], ignore_index=True)
    else:
        raise TypeError("Wrong type for benchmark_result")
