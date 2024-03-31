from typing import List, Sequence

import pandas as pds
from . import (
    BenchmarkResult,
    WorkflowResult,
    TaskDirResult,
    TrainTestSplitResult,
    task_dir_result_list_to_df,
    col_name_by_wastage_type,
    WastageTypes,
)


def order_models(evaluation_df: pds.DataFrame, cat: str) -> pds.DataFrame:
    return (
        evaluation_df.query(f"cat==@cat")
        .groupby(by=["short_name"])["val"]
        .mean()
        .reset_index()
        .sort_values(by=["val"])
    )


def evaluate_benchmark(
    benchmark: BenchmarkResult
    | WorkflowResult
    | TaskDirResult
    | Sequence[BenchmarkResult]
    | Sequence[WorkflowResult]
    | Sequence[TaskDirResult],
) -> pds.DataFrame:
    def flatten_one(one) -> List[TaskDirResult]:
        if isinstance(one, TaskDirResult):
            x_ = [one]
        else:
            x_ = one.flatten()
        return x_

    if isinstance(benchmark, Sequence):
        res = [y for bench in benchmark for y in flatten_one(bench)]
    else:
        res = flatten_one(benchmark)
    df = task_dir_result_list_to_df(res)
    return (
        df.query(f"cat=='{col_name_by_wastage_type[WastageTypes.Total]}'")
        .groupby(by=["orig", "perc", "retry_mode", "retry_model"])["val"]
        .mean()
        .reset_index()
    )
