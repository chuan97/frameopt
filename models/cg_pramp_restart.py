from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from frameopt.bounds import max_lower_bound
from frameopt.core.energy import coherence
from frameopt.core.frame import Frame
from frameopt.model.api import Problem, Result
from frameopt.model.p_scheduler import PScheduler
from frameopt.optim.local import cg_minimize


@dataclass(frozen=True, slots=True)
class CGPRampRestartModel:
    scheduler_factory: Callable[[], PScheduler]
    energy_func: Callable[[Frame, float], float]
    grad_func: Callable[[Frame, float], Any]
    seed: int
    maxiter: int = 1000
    step: int = 10
    restarts: int = 1

    @classmethod
    def from_config(cls, path: Path) -> CGPRampRestartModel:
        cfg = yaml.safe_load(path.read_text())
        init = cfg["init"]

        scfg = init.pop("scheduler")
        mod_name, _, class_name = scfg["import"].partition(":")
        mod = importlib.import_module(mod_name)
        sched_cls = getattr(mod, class_name)

        sinit = scfg["init"]
        if class_name == "AdaptivePScheduler" and "total_steps" not in sinit:
            sinit["total_steps"] = init["maxiter"] // init["step"]

        def factory() -> PScheduler:
            sch: PScheduler = sched_cls(**sinit)
            return sch

        init["scheduler_factory"] = factory

        ecfg = init.pop("energy")
        mod_name, _, func_name = ecfg["import"].partition(":")
        mod = importlib.import_module(mod_name)
        energy_func = getattr(mod, func_name)

        init["energy_func"] = energy_func

        gcfg = init.pop("grad")
        mod_name, _, func_name = gcfg["import"].partition(":")
        mod = importlib.import_module(mod_name)
        grad_func = getattr(mod, func_name)

        init["grad_func"] = grad_func

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

            rng_gen = np.random.default_rng(seed)
            step_best_frame = Frame.random(n=problem.n, d=problem.d, rng=rng_gen)

            best_frame = Frame.random(problem.n, problem.d)
            best_coh = coherence(best_frame)

            is_optimal = False

            for i in range(self.maxiter // self.step):

                def energy_fn(f: Frame, p: float = p) -> float:
                    return self.energy_func(f, p)

                def grad_fn(f: Frame, p: float = p) -> Any:
                    return self.grad_func(f, p)

                step_best_frame = cg_minimize(
                    frame0=step_best_frame,
                    energy_fn=energy_fn,
                    grad_fn=grad_fn,
                    maxiter=self.step,
                )
                step_best_coh = coherence(step_best_frame)

                if step_best_coh < best_coh:
                    best_coh = step_best_coh
                    best_frame = step_best_frame

                if best_coh < coh_lower_bound or math.isclose(
                    best_coh, coh_lower_bound, abs_tol=1e-9
                ):
                    is_optimal = True
                    break

                p, _ = scheduler.update(step=i, global_best_coh=best_coh)

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
