from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from evomof.bounds import max_lower_bound
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

    @classmethod
    def from_config(cls, path: Path) -> ProjectionModel:
        config = yaml.safe_load(path.read_text())
        init_dict = config["init"]

        return cls(**init_dict)

    def run(self, problem: Problem) -> Result:
        if problem.n <= problem.d:
            frame_vectors = np.eye(problem.d)[: problem.n, :]
            frame = Frame(frame_vectors)

            return Result(
                problem=problem,
                best_frame=frame,
                best_coherence=coherence(frame),
                wall_time_s=0.0,
            )

        coh_lower_bound = max_lower_bound(d=problem.d, n=problem.n)

        solver = ProjectionCMA(
            n=problem.n,
            d=problem.d,
            sigma0=self.sigma0,
            popsize=self.popsize,
            seed=self.seed,
            energy_fn=diff_coherence,
            energy_kwargs={"p": self.p},
        )

        best_frame = Frame.random(problem.n, problem.d)
        best_coh = coherence(best_frame)

        is_optimal = False

        t0 = time.perf_counter()
        for _ in range(1, self.max_gen + 1):
            gen_best_frame, _ = solver.step()
            gen_best_coh = coherence(gen_best_frame)

            if gen_best_coh < best_coh:
                best_coh = gen_best_coh
                best_frame = gen_best_frame

            if best_coh < coh_lower_bound or math.isclose(
                best_coh, coh_lower_bound, abs_tol=1e-10
            ):
                is_optimal = True
                break
        dt = time.perf_counter() - t0

        return Result(
            problem=problem,
            best_frame=best_frame,
            best_coherence=best_coh,
            wall_time_s=dt,
            extras={"optimal": is_optimal},
        )
