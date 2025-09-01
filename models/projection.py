from __future__ import annotations

import time

import numpy as np

from evomof.core.energy import coherence, diff_coherence
from evomof.core.frame import Frame
from evomof.optim.cma import ProjectionCMA
from models.api import Problem, Result


class ProjectionModel:
    def __init__(
        self,
        p: int = 2,
        sigma0: float = 0.5,
        popsize: int | None = None,
        max_gen: int = 1000,
        seed: int | None = None,
        log_every: int = 0,
    ):
        self.p = p
        self.sigma0 = sigma0
        self.popsize = popsize
        self.max_gen = max_gen
        self.seed = seed
        self.log_every = log_every

    @property
    def name(self) -> str:
        return "projection-cma/pure"

    def run(self, problem: Problem) -> Result:
        rng = np.random.default_rng(self.seed)
        start_frame = Frame.random(problem.n, problem.d, rng=rng)

        solver = ProjectionCMA(
            n=problem.n,
            d=problem.d,
            sigma0=self.sigma0,
            start_frame=start_frame,
            popsize=self.popsize,
            seed=self.seed,
            energy_fn=diff_coherence,
            energy_kwargs={"p": self.p},
        )

        t0 = time.perf_counter()
        best_frame = solver.run(max_gen=self.max_gen, log_every=self.log_every)
        dt = time.perf_counter() - t0

        return Result(
            best_frame=best_frame,
            best_coherence=float(coherence(best_frame)),
            wall_time_s=dt,
        )
