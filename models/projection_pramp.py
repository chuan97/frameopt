from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from frameopt.bounds import max_lower_bound
from frameopt.core.energy import coherence, pnormmax_coherence
from frameopt.core.frame import Frame
from frameopt.model.api import Problem, Result
from frameopt.model.p_scheduler import PScheduler
from frameopt.optim.cma import ProjectionCMA
from frameopt.optim.cma.utils import realvec_to_frame


@dataclass(frozen=True, slots=True)
class ProjectionPRampModel:
    scheduler_factory: Callable[[], PScheduler]
    sigma0: float = 0.3
    popsize: int | None = None
    max_gen: int = 50_000
    seed: int | None = None

    @classmethod
    def from_config(cls, path: Path) -> ProjectionPRampModel:
        cfg = yaml.safe_load(path.read_text())
        init = cfg["init"]

        scfg = init.pop("scheduler")
        mod_name, _, class_name = scfg["import"].partition(":")
        mod = importlib.import_module(mod_name)
        sched_cls = getattr(mod, class_name)

        sinit = scfg["init"]
        if class_name == "AdaptivePScheduler" and "total_steps" not in sinit:
            sinit["total_steps"] = init["max_gen"]

        def factory() -> PScheduler:
            sch: PScheduler = sched_cls(**sinit)
            return sch

        init["scheduler_factory"] = factory

        return cls(**init)

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

        scheduler = self.scheduler_factory()
        p = scheduler.current_p()

        coh_lower_bound = max_lower_bound(d=problem.d, n=problem.n)

        cma = ProjectionCMA(
            n=problem.n,
            d=problem.d,
            sigma0=self.sigma0,
            popsize=self.popsize,
            seed=self.seed,
        )

        best_frame = Frame.random(problem.n, problem.d)
        best_coh = coherence(best_frame)

        is_optimal = False

        t0 = time.perf_counter()
        for g in range(1, self.max_gen + 1):
            raws = cma.ask()
            frames = [realvec_to_frame(x, problem.n, problem.d) for x in raws]
            energies = [pnormmax_coherence(fr, p=p) for fr in frames]

            idx = int(np.argmin(energies))
            gen_best_frame = frames[idx]
            gen_best_coh = coherence(gen_best_frame)

            if gen_best_coh < best_coh:
                best_coh = gen_best_coh
                best_frame = gen_best_frame

            if best_coh < coh_lower_bound or math.isclose(
                best_coh, coh_lower_bound, abs_tol=1e-9
            ):
                is_optimal = True
                break

            cma.tell(raws, energies)
            p, _ = scheduler.update(step=g, global_best_coh=best_coh)
        dt = time.perf_counter() - t0

        return Result(
            problem=problem,
            best_frame=best_frame,
            best_coherence=best_coh,
            wall_time_s=dt,
            extras={"optimal": is_optimal},
        )
