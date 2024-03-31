import warnings
from dataclasses import dataclass
from typing import List

import numpy as np

from . import TrainTestSplitResult, simulate
from .. import only_assert_if_debug, FailedTrainingWarning, Globals, timed_function
from ..files import TaskTimeseriesCSVFileList
from ..models import (
    TaskModel,
    RetryMode,
    RetryModel,
    DefaultRetryModel,
    NoCheatDefaultTaskModel,
)
from ..progress import progress


@timed_function()
def benchmark_train_test_split(
    models: List[TaskModel],
    training: TaskTimeseriesCSVFileList,
    test: TaskTimeseriesCSVFileList,
    retry_mode: RetryMode,
    retry_model: RetryModel = None,
) -> List[TrainTestSplitResult]:
    only_assert_if_debug(
        training.wf_name == test.wf_name and training.task_name == test.task_name
    )
    if retry_model is None:
        retry_model = DefaultRetryModel()
    simulations = []
    for model in models:
        try:
            m_ = model.train(training)
        except:
            warnings.warn(
                f"{model.name} failed to train, falling back to DefaultTaskModel",
                FailedTrainingWarning,
            )
            m_ = NoCheatDefaultTaskModel().train(training)
        res = [
            simulate(
                m_,
                test=test_file,
                retry_mode=retry_mode,
                retry_model=retry_model,
                max_mem=Globals.max_mem,
            )
            for test_file in test
        ]
        simulations.append(
            TrainTestSplitResult(
                *np.average(res, axis=0),
                retry_mode=retry_mode,
                retry_model=retry_model,
                model=model,
                task_name=training.task_name,
                wf_name=training.wf_name,
            )
        )
        if not Globals.do_parallel:
            progress.advance()
    return simulations


@dataclass
class PoolBenchmarkTrainTestSplitArgs:
    models: List[TaskModel]
    training: TaskTimeseriesCSVFileList
    test: TaskTimeseriesCSVFileList
    retry_mode: RetryMode
    retry_model: RetryModel = None
    percentage: float = 0


def pool_benchmark_train_test_split(
    args: PoolBenchmarkTrainTestSplitArgs,
) -> tuple[list[TrainTestSplitResult], PoolBenchmarkTrainTestSplitArgs]:
    return (
        benchmark_train_test_split(
            models=args.models,
            training=args.training,
            test=args.test,
            retry_mode=args.retry_mode,
            retry_model=args.retry_model,
        ),
        args,
    )
