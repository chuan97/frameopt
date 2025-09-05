from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from evomof.core.energy import coherence, diff_coherence
from evomof.core.frame import Frame
from evomof.optim.cma import ProjectionCMA
from models.api import Problem, Result


@dataclass(frozen=True, slots=True)
class ProjectionModel:
    p: int = 2
    sigma0: float = 0.3
    popsize: int | None = None
    max_gen: int = 1000
    seed: int | None = None
    log_every: int = 0

    @property
    def name(self) -> str:
        return "projection-cma/pure"

    @classmethod
    def from_config(cls, path: Path) -> ProjectionModel:
        config = yaml.safe_load(path.read_text())
        init_dict = config["init"]

        return cls(**init_dict)

    def run(self, problem: Problem) -> Result:
        if problem.n <= problem.d:
            frame_vectors = np.eye(problem.d)[: problem.n, :]
            frame = Frame.from_array(frame_vectors)

            return Result(
                problem=problem,
                best_frame=frame,
                best_coherence=float(coherence(frame)),
                wall_time_s=0.0,
            )

        solver = ProjectionCMA(
            n=problem.n,
            d=problem.d,
            sigma0=self.sigma0,
            popsize=self.popsize,
            seed=self.seed,
            energy_fn=diff_coherence,
            energy_kwargs={"p": self.p},
        )

        t0 = time.perf_counter()
        best_frame = solver.run(
            max_gen=self.max_gen, log_every=self.log_every, tol=1e-20
        )
        dt = time.perf_counter() - t0

        return Result(
            problem=problem,
            best_frame=best_frame,
            best_coherence=float(coherence(best_frame)),
            wall_time_s=dt,
        )
