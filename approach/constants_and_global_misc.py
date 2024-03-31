import time
import warnings
from datetime import timedelta
from functools import wraps, cache
from pathlib import Path
from typing import Any, List, Sequence, Callable, TypeVar, Dict, Tuple

import dill
import numpy as np
import rich.pretty
import rich.traceback
from dask.distributed import Client, LocalCluster
from rich.console import Console
from sklearn.exceptions import ConvergenceWarning

from . import SegmentEdgePair, FailedTrainingWarning

rc = Console()
rich.pretty.install(console=rc, indent_guides=True, max_length=3, expand_all=True)
rich.traceback.install(console=rc, show_locals=False)


# constants
class Globals:
    debug = False
    do_parallel = False
    data_files: "DataDirectory" = None
    raise_failed_training = False
    suppress_failed_training = True
    show_function_runtimes = True
    base_data_path = Path(f"{__file__}/../../data").resolve().absolute()
    base_cache_dir = Path(f"{__file__}/../../cache").resolve().absolute()
    collection_interval = 2  # determined by the data collection and assumed to be constant across and within all time series
    max_mem = 128
    default_min_num_timeseries: int = 4
    default_min_num_data_points: int = 30
    default_timed_block_threshold: float = 0.5
    default_percentages: List[float] = [0.4, 0.5, 0.6, 0.7, 0.8]
    max_model_simulation_retries_per_minute: int = 6
    do_optimistic_smac_params: bool = False
    benchmark_task_dir_wall_time_limit: int = 0  # timedelta(minutes=2).total_seconds()
    benchmark_data_wall_time_limit: int = timedelta(minutes=15).total_seconds()
    use_pynisher: bool = True


if Globals.raise_failed_training:
    warnings.filterwarnings("error", category=FailedTrainingWarning)
if Globals.suppress_failed_training and not Globals.raise_failed_training:
    warnings.filterwarnings("ignore", category=FailedTrainingWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
Globals.do_parallel = (
    Globals.do_parallel and not Globals.debug
)  # can't debug when running parallel, so this flag should be overridden by debug
Globals.show_function_runtimes = (
    Globals.show_function_runtimes and not Globals.do_parallel
)
Globals.show_function_runtimes = False


def dasker() -> Callable[[], Client]:
    client = None

    def make() -> Client:
        nonlocal client
        if client is None:
            cluster = LocalCluster(
                # n_workers=16,
                threads_per_worker=1  # , memory_limit="5 GiB"
            )
            client = Client(cluster)
            rc.log(client.dashboard_link)
        return client

    return make


dask_client = dasker()


def only_assert_if_debug(condition: bool, message: str = None):
    if Globals.debug:
        assert condition, message
    else:
        pass


only_assert_if_debug(
    Globals.base_data_path.exists() and Globals.base_data_path.is_dir()
), f"Wrong base data path, {Globals.base_data_path.as_posix()}"


def transpose_uneven(
    lists: Sequence[Sequence[Any]],
) -> List[List[Any]]:  # todo function name is kind of unfitting
    # be aware that this operation is not invertible like common transposition would be
    ll = list(filter(lambda x: len(x) > 0, lists))
    max_len = max(len(x) for x in ll)
    min_len = min(len(x) for x in ll)
    o = []
    while len(ll):
        o.append([x.pop(0) for x in ll])
        ll = list(filter(lambda x: len(x) > 0, ll))
    return o


def unzip(l_: Sequence[Any]) -> List[Any]:
    return list(list(x) for x in zip(*l_))


def to_start_indices(run_lengths: Sequence[int]) -> np.ndarray:
    # converts run_lengths into list of start indices of segments
    o = []
    c = 0
    for r in run_lengths:
        o.append(c)
        c += r
    return np.asarray(o)


def from_start_indices(start_indices: Sequence[int]) -> np.ndarray:
    only_assert_if_debug(start_indices[0] == 0)
    run_lengths = np.diff(start_indices)
    only_assert_if_debug(all(run_lengths > 0))
    return run_lengths


def from_end_indices(end_indices: Sequence[int]) -> np.ndarray[int]:
    only_assert_if_debug(end_indices[0] != 0)
    end_indices = np.append([0], end_indices)
    run_lengths = np.diff(end_indices)
    only_assert_if_debug(all(run_lengths > 0))
    return run_lengths


def to_edge_indices(run_lengths: Sequence[int]) -> np.ndarray:
    # like to_start_indices but also includes total run length at the end
    return np.append(to_start_indices(run_lengths), [sum(run_lengths)])


def to_edge_pairs(run_lengths: Sequence[int]) -> List[SegmentEdgePair]:
    edges = to_edge_indices(run_lengths)
    return [SegmentEdgePair(s, e) for s, e in zip(edges[:-1], edges[1:])]


def to_segments(
    x: Sequence[float], run_lengths: Sequence[int]
) -> List[np.ndarray[float]]:
    # np.ndarray doesn't support a list of lists where inner lists have different lengths, so outer list is a list and not a np.ndarray
    return [np.array(x[p.start : p.end]) for p in to_edge_pairs(run_lengths)]


T = TypeVar("T")


execution_counts: Dict[str, int] = {}


def count_executions(name: str = None) -> Callable[..., T]:
    def decorator(f: Callable[..., T], /) -> Callable[..., T]:
        execution_counts[name] = 0

        @wraps(f)
        def wrapper(*args, **kwargs):
            execution_counts[name] += 1
            return f(*args, **kwargs)

        return wrapper

    return decorator


def timed_function(
    name: str = None,
    threshold: float = Globals.default_timed_block_threshold,
    enabled: bool = None,
) -> Callable[..., T]:
    if enabled is None:
        enabled = Globals.show_function_runtimes
    else:
        enabled = enabled

    def decorator(f: Callable[..., T], /) -> Callable[..., T]:
        if name is None:
            n_ = f"{f.__name__}"
        else:
            n_ = name

        @wraps(f)
        def wrapper(*args, **kwargs):
            with TimedBlock(n_, threshold=threshold, enabled=enabled):
                return f(*args, **kwargs)

        return wrapper

    return decorator


class TimedBlock:
    max_name: int = 0

    def __init__(
        self,
        name: str = "Timed",
        threshold: float = Globals.default_timed_block_threshold,
        enabled: bool = None,
    ):
        self.name = name
        self.threshold = threshold
        if enabled is None:
            self.enabled = Globals.show_function_runtimes
        else:
            self.enabled = enabled
        TimedBlock.max_name = max(TimedBlock.max_name, len(name))

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            passed = timedelta(seconds=time.perf_counter() - self.start)
            if (
                passed.seconds >= self.threshold
                or passed.microseconds >= self.threshold * 1_000_000
            ):
                if Globals.do_parallel:
                    printer = print
                else:
                    printer = rc.print
                printer(
                    f"{f'[ {self.name} ]':<{TimedBlock.max_name + 4}} took {passed}"
                )


def cached_to_file(
    *distinguishers, dir_path: Path = Globals.base_cache_dir, f_name: str = None
):
    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @wraps(f)
        def wrapper(*args, **kwargs):
            d_: Path = dir_path
            d_ = d_.joinpath(f_name if f_name is not None else f.__name__)
            for d in distinguishers:
                d_ = d_.joinpath(str(d))
            for a in args:
                d_ = d_.joinpath(str(a))
            for kw in kwargs:
                d_ = d_.joinpath(str(kw))
            f_ = d_.joinpath("dill.pickle")
            if f_.exists():
                with open(f_, "rb") as file:
                    return dill.load(file)
            else:
                res = f(*args, **kwargs)
                d_.mkdir(parents=True, exist_ok=True)
                with open(f_, "wb") as file:
                    dill.dump(res, file)
                return res

        return wrapper

    return decorator


def name_shortener() -> (
    tuple[Callable[[str], str], Callable[[], None], Callable[[], None]]
):
    shortened_names: List[str] = []

    @cache
    def shorten(name: str):
        parts = name.split("(", 1)
        if len(parts) > 1:
            if name not in shortened_names:
                shortened_names.append(name)
            return f"{parts[0]} {shortened_names.index(name)}"
        else:
            return name

    def reset():
        nonlocal shortened_names
        shortened_names = []

    def show():
        for name in shortened_names:
            rc.print(f"{shorten(name)} == {name}")

    return shorten, reset, show


shorten_name, reset_shortened_names, show_shortened_names = name_shortener()
