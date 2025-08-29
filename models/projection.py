from __future__ import annotations

import time

import numpy as np

from evomof.core.energy import coherence, diff_coherence
from evomof.core.frame import Frame
from evomof.optim.cma import ProjectionCMA
from models.api import Problem, Result
from models.utils import count_calls


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

        @count_calls
        def objective(frame: Frame) -> float:
            return diff_coherence(frame, p=self.p)

        solver = ProjectionCMA(
            n=problem.n,
            d=problem.d,
            sigma0=self.sigma0,
            start_frame=start_frame,
            popsize=self.popsize,
            seed=self.seed,
            energy_fn=objective,
        )

        t0 = time.perf_counter()
        best_frame = solver.run(max_gen=self.max_gen, log_every=self.log_every)
        dt = time.perf_counter() - t0

        n_calls = getattr(objective, "n_calls", 0)

        return Result(
            best_frame=best_frame,
            best_coherence=float(coherence(best_frame)),
            n_calls=n_calls,
            wall_time_s=dt,
            extras={
                "train_objective": "diff_coherence",
                "train_objective_params": {"p": self.p},
                "sigma0": self.sigma0,
                "popsize": self.popsize,
                "total_evals": self.max_gen,
            },
        )
