from datetime import timedelta

import rich.progress
from rich.progress import Progress as rProgress
from rich.text import Text

from .. import rc, only_assert_if_debug


# progress display
def round_to_first_significant_digits(num, digits=1, max=3):
    only_assert_if_debug(digits >= 1)
    only_assert_if_debug(max >= 0)
    if round(num, max) == 0:
        return 0.0
    #
    firstSigDigit = 0
    while round(num, firstSigDigit) == 0.0:
        firstSigDigit += 1
    roundTo = firstSigDigit + digits - 1
    roundTo = max if roundTo > max else roundTo
    return round(num, roundTo)


def ema(x, mu=None, alpha=0.3):
    # taken from https://github.com/timwedde/rich-utils/blob/master/rich_utils/progress.py
    """
    Exponential moving average: smoothing to give progressively lower
    weights to older values.
    Parameters
    ----------
    x  : float
        New value to include in EMA.
    mu  : float, optional
        Previous EMA value.
    alpha  : float, optional
        Smoothing factor in range [0, 1], [default: 0.3].
        Increase to give more weight to recent values.
        Ranges from 0 (yields mu) to 1 (yields x).
    """
    return x if mu is None else (alpha * x) + (1 - alpha) * mu


class ItemsPerSecondColumn(rich.progress.ProgressColumn):
    max_refresh = 1

    def __init__(self):
        super().__init__()
        self.seen = dict()
        self.itemsPS = dict()

    def render(self, task: "rich.progress.Task") -> Text:
        if task.id not in self.seen.keys():
            self.seen[task.id] = 0
            self.itemsPS[task.id] = 0.0
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            self.seen[task.id] = 0
            self.itemsPS[task.id] = 0.0
            return Text("(0.0/s)", style="progress.elapsed")
        if task.finished:
            return Text(
                f"({round_to_first_significant_digits(task.completed / elapsed, 3, 3):,}/s)",
                style="progress.elapsed",
            )
        if self.seen[task.id] > task.completed:
            self.seen[task.id] = 0
        if self.seen[task.id] == 0 and task.completed > 0:
            self.itemsPS[task.id] = round_to_first_significant_digits(
                task.completed / elapsed, 3, 3
            )
        if True:  # self.seen[task.id] < task.completed:
            self.itemsPS[task.id] = round_to_first_significant_digits(
                ema(
                    round_to_first_significant_digits(task.completed / elapsed, 3, 3),
                    self.itemsPS[task.id],
                ),
                3,
                3,
            )
            self.seen[task.id] = task.completed
        return Text(f"({self.itemsPS[task.id]:,}/s)", style="progress.elapsed")


class SecondsPerItemColumn(rich.progress.ProgressColumn):
    max_refresh = 1

    def __init__(self):
        super().__init__()
        self.seen = dict()
        self.secPerItem = dict()

    def render(self, task: "rich.progress.Task") -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            self.seen[task.id] = 0
            self.secPerItem[task.id] = 0.0
            return Text("(0.0s/item)", style="progress.elapsed")
        if task.finished:
            trueSPI = round_to_first_significant_digits(elapsed / task.completed, 3, 3)
            if trueSPI <= 60:
                return Text(
                    f"({trueSPI}s/item)",
                    style="progress.elapsed",
                )
            else:
                return Text(
                    f"({timedelta(seconds=int(trueSPI))}/item)",
                    style="progress.elapsed",
                )
        #
        if task.completed == 0:
            self.seen[task.id] = 0
            self.secPerItem[task.id] = round_to_first_significant_digits(elapsed, 3, 3)
            if self.secPerItem[task.id] <= 60:
                return Text(
                    f"({self.secPerItem[task.id]}s/item)", style="progress.elapsed"
                )
            else:
                return Text(
                    f"({timedelta(seconds=int(self.secPerItem[task.id]))}/item)",
                    style="progress.elapsed",
                )
        #
        if True:  # self.seen[task.id] < task.completed:
            self.secPerItem[task.id] = round_to_first_significant_digits(
                ema(
                    round_to_first_significant_digits(elapsed / task.completed, 3, 3),
                    self.secPerItem[task.id],
                ),
                3,
                3,
            )
            self.seen[task.id] = task.completed
        if self.secPerItem[task.id] <= 60:
            return Text(f"({self.secPerItem[task.id]}s/item)", style="progress.elapsed")
        else:
            return Text(
                f"({timedelta(seconds=int(self.secPerItem[task.id]))}/item)",
                style="progress.elapsed",
            )


# taken from https://github.com/timwedde/rich-utils/blob/master/rich_utils/progress.py
class SmartTimeRemainingColumn(rich.progress.ProgressColumn):
    max_refresh = 1

    def __init__(self, *args, **kwargs):
        self.seen = dict()
        self.avg_remaining_seconds = dict()
        self.smoothing = kwargs.pop("smoothing", 0.3)
        super().__init__(*args, **kwargs)

    def render(self, task):
        if task.finished:
            return Text("-:--:--", style="progress.remaining")
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            self.seen[task.id] = 0
            self.avg_remaining_seconds[task.id] = 0.0
            return Text("-:--:--", style="progress.remaining")
        if task.completed == 0:
            self.seen[task.id] = 0
            self.avg_remaining_seconds[task.id] = 0.0
            return Text("-:--:--", style="progress.remaining")
        speed = elapsed / task.completed
        remaining = (task.total - task.completed) * speed
        if self.seen[task.id] > task.completed:
            self.seen[task.id] = 0
            self.avg_remaining_seconds[task.id] = remaining
        #
        if True:  # self.seen[task.id] < task.completed:
            self.avg_remaining_seconds[task.id] = ema(
                remaining, self.avg_remaining_seconds[task.id], self.smoothing
            )
            self.seen[task.id] = task.completed
        return Text(
            str(timedelta(seconds=int(self.avg_remaining_seconds[task.id]))),
            style="progress.remaining",
        )


def std_progress(console=None, *args, **kwargs):
    if console is None:
        console = rc
    return rProgress(
        "[progress.description]{task.description}",
        rich.progress.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "({task.completed}/{task.total})",
        rich.progress.TimeElapsedColumn(),
        "eta:",
        SmartTimeRemainingColumn(),
        ItemsPerSecondColumn(),
        SecondsPerItemColumn(),
        *args,
        **kwargs,
        console=console,
        transient=True,
    )


class Progress:
    progress = None
    prog_id = None

    def __init__(self):
        self.progress = std_progress(rc, refresh_per_second=10)

    def get_progress(self):
        return self.progress

    def add_prog_bar(self, name: str, total: int):
        self.prog_id = self.progress.add_task(name, total=total)
        return self.prog_id

    def advance(self):
        if self.prog_id is not None:
            self.progress.advance(self.prog_id)

    def reset(self):
        self.progress = std_progress(rc, refresh_per_second=10)
        self.prog_id = None


progress = Progress()
