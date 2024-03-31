import enum
from enum import Enum
from typing import NamedTuple, List

from ..models import RetryMode, RetryModel, BaseModel


class WastageTypes(Enum):
    Over = enum.auto()
    Under = enum.auto()
    Total = enum.auto()


class Waste:
    over_waste: float
    under_waste: float
    total: float

    def __init__(self, over_waste: float = 0, under_waste: float = 0):
        self.over_waste = over_waste
        self.under_waste = under_waste

    @property
    def over(self):
        return self.over_waste

    @property
    def under(self):
        return self.under_waste

    @property
    def total(self) -> float:
        return self.over_waste + self.under_waste

    def add_under(self, under: float) -> float:
        self.under_waste += under
        return self.under_waste

    def add_over(self, over: float) -> float:
        self.over_waste += over
        return self.over_waste

    def add(self, val: float, over: bool) -> float:
        if over:
            self.add_over(val)
            return self.over_waste
        else:
            self.add_under(val)
            return self.under_waste

    def __add__(self, other):
        if isinstance(other, Waste):
            return Waste(
                self.over_waste + other.over_waste, self.under_waste + other.under_waste
            )
        else:
            raise TypeError("Can only add Waste objects")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Waste(self.over_waste * other, self.under_waste * other)
        else:
            raise TypeError("Can only multiply with scalar numbers")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Waste(self.over_waste / other, self.under_waste / other)
        else:
            raise TypeError("Can only divide by scalar numbers")


class TrainTestSplitResult(NamedTuple):
    avg_waste: Waste
    avg_retries: float
    avg_runtime: float
    avg_efficacy: float
    retry_mode: RetryMode
    retry_model: RetryModel
    model: BaseModel
    task_name: str
    wf_name: str


class TaskDirResult(NamedTuple):
    split_results: List[TrainTestSplitResult]
    retry_mode: RetryMode
    retry_model: RetryModel
    percentage: float
    task_name: str
    wf_name: str


class WorkflowResult(NamedTuple):
    task_dir_results: List[TaskDirResult]
    task_name: str
    retry_mode: RetryMode
    retry_model: RetryModel
    percentages: List[float]
    wf_name: str

    def flatten(self) -> List[TaskDirResult]:
        return [task_dir for task_dir in self.task_dir_results]


class BenchmarkResult(NamedTuple):
    workflow_results: List[WorkflowResult]
    wf_name: str
    retry_mode: RetryMode
    retry_model: RetryModel
    percentages: List[float]

    def flatten(self) -> List[TaskDirResult]:
        return [res for wf in self.workflow_results for res in wf.flatten()]


class AllocationCheckResult(NamedTuple):
    success: bool
    waste: Waste
    failed_at: int


class SimulationResult(NamedTuple):
    waste: Waste
    retries: int
    runtime: int
    efficacy: float
