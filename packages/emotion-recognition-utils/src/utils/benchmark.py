from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any


class Timer:
    """用于测量代码块执行时间的上下文管理器类"""

    def __init__(
        self,
        callback: Callable[[float], Any] | None = None,
    ):
        if callback is None:
            self.callback = lambda x: print(f"代码块执行耗时: {x} 秒")
        else:
            self.callback = callback
        self.start_time: float | None = None
        self.end_time: float | None = None

    def __enter__(self):
        if self.start_time is not None:
            raise RuntimeError("Timer already started")
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.start_time is not None:
            execution_time = self.end_time - self.start_time
            self.callback(execution_time)

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


@contextmanager
def timer_context_manager(template: str = "代码块执行耗时: {} 秒"):
    def formatted_print_callback(time_elapsed):
        return print(template.format(time_elapsed))

    timer = Timer(callback=formatted_print_callback)
    with timer:
        yield
