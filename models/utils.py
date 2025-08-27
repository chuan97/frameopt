from __future__ import annotations

from collections.abc import Callable
from functools import update_wrapper
from typing import Generic, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class CallCounter(Generic[P, R]):
    """
    A lightweight, type-safe callable wrapper that counts how many times it's called.

    Usage:
        def f(x: int) -> int: ...
        counted = CallCounter(f)
        y = counted(3)
        assert counted.n_calls == 1

    Notes:
    - We keep the original function's metadata (name, docstring) via `update_wrapper`.
    - `reset()` lets you zero the counter between runs if you re-use the same instance.
    """

    def __init__(self, fn: Callable[P, R]) -> None:
        self._fn = fn
        self.n_calls: int = 0
        update_wrapper(self, fn)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        self.n_calls += 1
        return self._fn(*args, **kwargs)

    def reset(self) -> None:
        self.n_calls = 0


def count_calls(fn: Callable[P, R]) -> CallCounter[P, R]:
    """
    Decorator/adapter that returns a `CallCounter` wrapper for `fn`.

    Example:
        @count_calls
        def energy(F: Frame) -> float:
            ...

        # or, without decorator syntax:
        energy = count_calls(energy)

    The returned object is callable and exposes:
        - `.n_calls: int`  — number of invocations
        - `.reset() -> None`
    """
    return CallCounter(fn)
