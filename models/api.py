from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from evomof.core.frame import Frame


@dataclass(frozen=True)
class Problem:
    d: int
    n: int


@dataclass
class Result:
    best_frame: Frame
    best_coherence: float
    wall_time_s: float
    extras: dict[str, Any] | None = None


class Model(Protocol):
    @property
    def name(self) -> str: ...
    def run(self, problem: Problem) -> Result: ...
