from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from frameopt.core.frame import Frame

__all__ = ["Problem", "Result", "Model"]


@dataclass(frozen=True, slots=True)
class Problem:
    d: int
    n: int


@dataclass
class Result:
    problem: Problem
    best_frame: Frame
    best_coherence: float
    wall_time_s: float
    extras: dict[str, Any] = field(default_factory=dict)


class Model(Protocol):
    def run(self, problem: Problem) -> Result: ...

    @classmethod
    def from_config(cls, path: str) -> Model: ...
