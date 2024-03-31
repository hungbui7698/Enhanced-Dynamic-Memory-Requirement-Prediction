from functools import cache

from . import RetryModel, RetryMode


class BufferRetryModel(RetryModel):
    name = "BufferRetryModel"

    def __init__(self, percent: float = 1, min_: float = 10.0):
        self.percent = percent
        self.min_ = min_

    def predict(
        self, usage: float, allocation: float, retry_mode: RetryMode, *args, **kwargs
    ) -> float:
        target = usage * (1 + self.percent)
        target = max(target, usage + self.min_)
        return target / allocation

    @cache
    def __repr__(self):
        return f"{self.name}({self.percent}, {self.min_})"


class ReflectRetryModel(RetryModel):
    name = "ReflectRetryModel"

    def __init__(self, factor: float = 1, min_percent: float = 1):
        self.factor = factor
        self.min_percent = min_percent

    def predict(
        self, usage: float, allocation: float, retry_mode: RetryMode, *args, **kwargs
    ) -> float:
        target = usage + self.factor * (usage - allocation)
        return max(target / allocation, 1 + self.min_percent)

    @cache
    def __repr__(self):
        return f"{self.name}({self.factor}, {self.min_percent})"
