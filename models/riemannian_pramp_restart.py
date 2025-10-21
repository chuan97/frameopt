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
from frameopt.core.energy import coherence
from frameopt.core.frame import Frame
from frameopt.model.api import Problem, Result
from frameopt.model.p_scheduler import PScheduler
from frameopt.optim.cma import RiemannianCMA


@dataclass(frozen=True, slots=True)
class RiemannianPRampRestartModel:
    scheduler_factory: Callable[[], PScheduler]
    energy_func: Callable[[Frame, float], float]
    seed: int
    sigma0: float = 0.3
    popsize: int | None = None
    max_gen: int = 50_000
    restarts: int = 1

    @classmethod
    def from_config(cls, path: Path) -> RiemannianPRampRestartModel:
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

        ecfg = init.pop("energy")
        mod_name, _, func_name = ecfg["import"].partition(":")
        mod = importlib.import_module(mod_name)
        energy_func = getattr(mod, func_name)

        init["energy_func"] = energy_func

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

        coh_lower_bound = max_lower_bound(d=problem.d, n=problem.n)
        overall_best_frame = Frame.random(problem.n, problem.d)
        overall_best_coh = coherence(overall_best_frame)
        seed: int = self.seed

        t0 = time.perf_counter()
        for _ in range(self.restarts):
            scheduler = self.scheduler_factory()
            p = scheduler.current_p()

            cma = RiemannianCMA(
                n=problem.n,
                d=problem.d,
                sigma0=self.sigma0,
                popsize=self.popsize,
                seed=seed,
            )

            best_frame = Frame.random(problem.n, problem.d)
            best_coh = coherence(best_frame)

            is_optimal = False

            for g in range(1, self.max_gen + 1):
                frames = cma.ask()
                energies = np.array([self.energy_func(fr, p) for fr in frames])

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

                cma.tell(frames, energies)
                p, _ = scheduler.update(step=g, global_best_coh=best_coh)

            if best_coh < overall_best_coh:
                overall_best_coh = best_coh
                overall_best_frame = best_frame

            if is_optimal:
                overall_best_frame = best_frame
                overall_best_coh = best_coh
                break

            seed += 1
        dt = time.perf_counter() - t0

        return Result(
            problem=problem,
            best_frame=overall_best_frame,
            best_coherence=overall_best_coh,
            wall_time_s=dt,
            extras={"optimal": is_optimal},
        )
