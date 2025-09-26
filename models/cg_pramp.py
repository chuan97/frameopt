from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import yaml

from frameopt.bounds import max_lower_bound
from frameopt.core.energy import coherence, diff_coherence, grad_diff_coherence
from frameopt.core.frame import Frame
from frameopt.models.api import Problem, Result
from frameopt.models.p_scheduler import PScheduler
from frameopt.optim.local import cg_minimize


@dataclass(frozen=True, slots=True)
class CGPRampModel:
    scheduler_factory: Callable[[], PScheduler]
    maxiter: int = 1000
    seed: int | None = None
    step: int = 10

    @classmethod
    def from_config(cls, path: Path) -> CGPRampModel:
        cfg = yaml.safe_load(path.read_text())
        init = cfg["init"]

        scfg = init["scheduler"]
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
        init.pop("scheduler")

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

        rng_gen = np.random.default_rng(self.seed)
        step_best_frame = Frame.random(n=problem.n, d=problem.d, rng=rng_gen)

        best_frame = Frame.random(problem.n, problem.d)
        best_coh = coherence(best_frame)

        is_optimal = False

        t0 = time.perf_counter()
        for i in range(self.maxiter // self.step):
            step_best_frame = cg_minimize(
                frame0=step_best_frame,
                energy_fn=partial(diff_coherence, p=p),
                grad_fn=partial(grad_diff_coherence, p=p),
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
        dt = time.perf_counter() - t0

        return Result(
            problem=problem,
            best_frame=best_frame,
            best_coherence=best_coh,
            wall_time_s=dt,
            extras={"optimal": is_optimal},
        )
