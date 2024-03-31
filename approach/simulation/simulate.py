import time
from pathlib import Path
from typing import Sequence, List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from . import AllocationCheckResult, SimulationResult, Waste
from .. import (
    only_assert_if_debug,
    Globals,
)
from ..data_types import SegmentEdgePair
from ..files import (
    TaskTimeseriesCSVFiles,
    TaskTimeseriesCSVDirectory,
    TaskTimeseriesCSVFileList,
    DataDirectory,
    WFCSVDirectory,
)
from ..models import (
    DefaultTaskModel,
    TaskModel,
    RetryModel,
    DefaultRetryModel,
    RetryMode,
)


def selective_retry(
    allocation: np.ndarray[float],
    *,
    retry_factor: float,
    failed_at: int,
    segment_edges: Sequence[SegmentEdgePair] = None,
    max_mem: float = None,
) -> np.ndarray[float]:
    if segment_edges is None:
        c_ = allocation_to_segment_edges(allocation)
    else:
        c_ = segment_edges
    a_ = np.array(allocation)  # make copy
    if failed_at > c_[-1].end:
        a_[c_[-1].start : c_[-1].end + 1] = (
            a_[c_[-1].start : c_[-1].end + 1] * retry_factor
        )
        return a_
    for cluster in c_:
        if cluster.start <= failed_at <= cluster.end:
            a_[cluster.start : cluster.end + 1] = (
                a_[cluster.start : cluster.end + 1] * retry_factor
            )
            break
    return a_


def partial_retry(
    allocation: np.ndarray[float],
    *,
    retry_factor: float,
    failed_at: int,
    segment_edges: Sequence[SegmentEdgePair] = None,
    max_mem: float = None,
) -> np.ndarray[float]:
    if segment_edges is None:
        c_ = allocation_to_segment_edges(allocation)
    else:
        c_ = segment_edges
    a_ = np.array(allocation)  # make copy
    if failed_at > c_[-1].end:
        a_[c_[-1].start : c_[-1].end + 1] = (
            a_[c_[-1].start : c_[-1].end + 1] * retry_factor
        )
        return a_
    for cluster in c_:
        if cluster.end >= failed_at:
            a_[cluster.start : cluster.end + 1] = (
                a_[cluster.start : cluster.end + 1] * retry_factor
            )
    return a_


def full_retry(
    allocation: np.ndarray[float],
    *,
    retry_factor: float,
    failed_at: int,
    segment_edges: Sequence[SegmentEdgePair] = None,
    max_mem: float = None,
) -> np.ndarray[float]:
    return np.array(allocation) * retry_factor


def tovar_retry(
    allocation: np.ndarray[float],
    *,
    retry_factor: float,
    failed_at: int,
    segment_edges: Sequence[SegmentEdgePair] = None,
    max_mem: float = None,
) -> np.ndarray[float]:
    return np.ones(len(allocation)) * (max_mem * 1000)


RetryMode.retry_fn_lut = {
    RetryMode.Full: full_retry,
    RetryMode.Tovar: tovar_retry,
    RetryMode.Selective: selective_retry,
    RetryMode.Partial: partial_retry,
}


def best_case_wastage(usage: np.ndarray[float], allocation: np.ndarray[float]) -> Waste:
    c_ = allocation_to_segment_edges(allocation)
    assert isinstance(c_, list)
    assert all(isinstance(x, SegmentEdgePair) for x in c_)
    best_case = np.asarray(
        [
            x
            for c in c_
            for x in [max(usage[c.start : c.end + 1])] * (c.end + 1 - c.start)
        ]
    )
    return success_wastage(usage, best_case)


def success_wastage(usage: np.ndarray[float], allocation: np.ndarray[float]) -> Waste:
    only_assert_if_debug(
        np.all(allocation >= usage),
        "Only use this method when allocation doesnt fail (ie. is always larger than usage)",
    )
    return Waste(over_waste=np.trapz(allocation) - np.trapz(usage), under_waste=0)


def failure_wastage(
    usage: np.ndarray[float], allocation: np.ndarray[float], failed_at: int
) -> Waste:
    u_ = usage[: failed_at + 1]
    a_ = allocation[: failed_at + 1]
    only_assert_if_debug(
        np.allclose(
            np.trapz(np.fmax(a_ - u_, 0)) + np.trapz(u_),
            np.trapz(np.fmax(a_, u_)),
            atol=1e-4,
        ),
        f"{np.trapz(np.fmax(a_ - u_, 0)) + np.trapz(u_)} != {np.trapz(np.fmax(a_, u_))}",
    )
    return Waste(over_waste=np.trapz(np.fmax(a_ - u_, 0)), under_waste=np.trapz(u_))


def allocation_to_segment_edges(allocation: Sequence[float]) -> List[SegmentEdgePair]:
    only_assert_if_debug(len(allocation) >= 1)
    cur = allocation[0]
    last_i = 0
    clusters = []
    for i, v in enumerate(allocation):
        if cur != v:
            clusters.append(SegmentEdgePair(last_i, i - 1))
            cur = v
            last_i = i
    clusters.append(SegmentEdgePair(last_i, i))
    return clusters


def end_pad_allocation(usage, allocation) -> np.ndarray[float]:
    if len(allocation) < len(usage):
        a_: np.ndarray = np.append(
            allocation, [allocation[-1]] * (len(usage) - len(allocation))
        )
    else:
        a_ = allocation
    return a_[: len(usage)]


def check_allocation(
    usage: np.ndarray[float], allocation: np.ndarray[float]
) -> AllocationCheckResult:
    only_assert_if_debug(len(allocation) >= 1)
    only_assert_if_debug(len(usage) >= 1)
    only_assert_if_debug(
        len(allocation) == len(usage), f"{len(usage)=} != {len(allocation)=}"
    )

    failures: np.ndarray = usage > allocation
    failed_at: np.ndarray = failures.nonzero()[0]
    if len(failed_at) > 0:
        failed_at: int = failed_at[0]
        return AllocationCheckResult(
            False, failure_wastage(usage, allocation, failed_at), failed_at
        )
    else:
        # there were no failures
        return AllocationCheckResult(
            True,
            success_wastage(usage, allocation[: len(usage)]),
            len(usage) - 1,  # todo return len(usage) or len(usage) - 1 here?
        )


class BadModelException(Exception):
    pass


def failures_in_the_last_minute(failures: List[int]):
    if len(failures) < 2:
        return len(failures)
    last = failures[-1]
    window_start = -1
    for i, v in enumerate(failures):
        if last - 30 <= v <= last:
            window_start = i
            break
    else:
        return 1
    return len(failures) - window_start


def simulate(
    task_model: TaskModel,
    test: TaskTimeseriesCSVFiles,
    retry_mode: RetryMode,
    retry_model: RetryModel,
    max_mem: float,
) -> SimulationResult:
    memory, metadata = (
        test.memory,
        test.metadata,
    )  # memory, cpu, file_event, metadata
    usage = memory.values
    if isinstance(
        task_model, DefaultTaskModel
    ):  # todo move this logic to DefaultTaskModel
        allocation = np.ones(len(usage)) * metadata.default_value
    else:
        allocation = task_model.predict(metadata.input_total_size)
    only_assert_if_debug(
        len(allocation.shape) == 1
        or (len(allocation.shape) == 2 and allocation.shape[1] == 1)
    )
    allocation = end_pad_allocation(usage, allocation)
    waste = Waste()
    runtime = 0
    retries = 0
    success = False
    failures = []
    while not success:
        simulation = check_allocation(usage, allocation)
        success = simulation.success
        if Globals.debug and not success:
            compare_usage_and_allocation(
                usage,
                allocation,
                task_model.name,
                task_name=test.memory.csv_file.task_name,
            )
        waste += simulation.waste
        failed_at = simulation.failed_at
        runtime += simulation.failed_at + 1
        if success:
            break
        retries += 1
        failures.append(failed_at)
        if (
            failures_in_the_last_minute(failures)
            >= Globals.max_model_simulation_retries_per_minute
        ):
            raise BadModelException()
        allocation = RetryMode.retry_fn_lut[retry_mode](
            allocation=allocation,
            retry_factor=retry_model.predict(
                usage[failed_at],
                allocation[failed_at],
                retry_mode,
            ),
            failed_at=failed_at,
            max_mem=max_mem,
        )
    if Globals.debug and retries != 0:
        compare_usage_and_allocation(
            usage,
            allocation,
            task_model.name,
            task_name=test.memory.csv_file.task_name,
        )
    return SimulationResult(
        waste / 1000 * Globals.collection_interval,
        retries,
        runtime * Globals.collection_interval,
        best_case_wastage(usage, allocation).over
        / success_wastage(usage, allocation).over,
    )


# unused
class Simulation:
    task_model: TaskModel
    data: TaskTimeseriesCSVFileList
    retry_mode: RetryMode
    retry_model: RetryModel
    max_mem: float

    def __init__(
        self,
        task_model: TaskModel,
        data_dir: Path = None,
        data: TaskTimeseriesCSVFileList = None,
        retry_mode: RetryMode = RetryMode.Full,
        retry_model: RetryModel = None,
        max_mem: float = 128,  # in GB
    ):
        only_assert_if_debug(data_dir is not None or data is not None)
        if data is not None:
            self.data = data
        else:
            self.data = TaskTimeseriesCSVDirectory(data_dir).content
        self.task_model = task_model
        self.retry_model = retry_model
        if retry_model is None:
            self.retry_model = DefaultRetryModel()
        else:
            self.retry_mode = retry_mode
        self.max_mem = max_mem

    def simulate(self, task_csv_files: TaskTimeseriesCSVFiles) -> SimulationResult:
        return simulate(
            self.task_model,
            task_csv_files,
            self.retry_mode,
            self.retry_model,
            self.max_mem,
        )


def compare_usage_and_allocation(
    usage: Sequence, allocation: Sequence, model_name: str, task_name: str
) -> None:
    only_assert_if_debug(len(usage) == len(allocation))
    g = sns.lineplot(x=np.arange(len(usage)), y=usage)
    sns.lineplot(x=np.arange(len(allocation)), y=allocation, ax=g)
    g.set_title(f"{model_name} on {task_name}")
    time.sleep(0.5)
    plt.show()


def calc_total_work(
    files: DataDirectory | WFCSVDirectory,
    models: List[TaskModel],
    percentages: List[float],
    retry_modes: List[RetryMode],
    retry_models: List[RetryModel],
    shuffle_repeats: int,
) -> int:
    if isinstance(files, DataDirectory):
        num_task_dirs = sum(len(wf_dir_files.content) for wf_dir_files in files.content)
    else:
        num_task_dirs = len(files.content)
    if Globals.do_parallel:
        total_work = num_task_dirs * len(retry_modes) * len(retry_models)
    else:
        total_work = (
            num_task_dirs
            * len(percentages)
            * len(retry_modes)
            * len(retry_models)
            * shuffle_repeats
            * len(models)
        )
    return total_work
