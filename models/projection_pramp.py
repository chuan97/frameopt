from __future__ import annotations

# --- Added imports ---
import time

import numpy as np

from evomof.core.energy import coherence, diff_coherence
from evomof.core.frame import Frame
from evomof.optim.cma import ProjectionCMA
from evomof.optim.utils.p_scheduler import FixedPScheduler, Scheduler
from models.api import Problem, Result


class ProjectionPRampModel:
    def __init__(
        self,
        sigma0: float = 0.5,
        popsize: int | None = None,
        max_gen: int = 50_000,
        seed: int | None = None,
        log_every: int = 0,
        scheduler: Scheduler | None = None,
    ):
        self.sigma0 = sigma0
        self.popsize = popsize
        self.max_gen = max_gen
        self.seed = seed
        self.log_every = log_every

        self.scheduler: Scheduler
        if scheduler is None:
            self.scheduler = FixedPScheduler()
        else:
            self.scheduler = scheduler

    @property
    def name(self) -> str:
        return "projection-cma/pramp"

    def run(self, problem: Problem) -> Result:
        rng = np.random.default_rng(self.seed)
        start_frame = Frame.random(problem.n, problem.d, rng=rng)
        cma = ProjectionCMA(
            n=problem.n,
            d=problem.d,
            sigma0=self.sigma0,
            start_frame=start_frame,
            popsize=self.popsize,
            seed=self.seed,
        )
        p = self.scheduler.current_p()

        best_frame = start_frame
        best_coh = float(coherence(best_frame))

        t0 = time.perf_counter()
        for g in range(1, self.max_gen + 1):
            population = cma.ask()
            energies = [diff_coherence(F, p=p) for F in population]

            idx = int(np.argmin(energies))
            gen_best_frame = population[idx]
            gen_best_coh = float(coherence(gen_best_frame))

            if gen_best_coh < best_coh:
                best_coh = gen_best_coh
                best_frame = gen_best_frame

            cma.tell(population, energies)

            p, _ = self.scheduler.update(step=g, global_best_coh=best_coh)
        dt = time.perf_counter() - t0

        return Result(
            best_frame=best_frame,
            best_coherence=best_coh,
            wall_time_s=dt,
        )
