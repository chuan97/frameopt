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
from frameopt.core.energy import coherence
from frameopt.core.frame import Frame
from frameopt.model.api import Problem, Result
from frameopt.optim.cma import RiemannianCMA


@dataclass(frozen=True, slots=True)
class RiemannianModel:
    energy_func: Callable[[Frame], float]
    seed: int
    sigma0: float = 0.3
    popsize: int | None = None
    max_gen: int = 1000
    restarts: int = 1

    @classmethod
    def from_config(cls, path: Path) -> RiemannianModel:
        cfg = yaml.safe_load(path.read_text())
        init = cfg["init"]

        ecfg = init.pop("energy")
        mod_name, _, func_name = ecfg["import"].partition(":")
        mod = importlib.import_module(mod_name)
        energy_func = getattr(mod, func_name)
        energy_func = partial(energy_func, **ecfg["kwargs"])

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
            solver = RiemannianCMA(
                n=problem.n,
                d=problem.d,
                sigma0=self.sigma0,
                popsize=self.popsize,
                seed=seed,
                energy_fn=self.energy_func,
            )

            best_frame = Frame.random(problem.n, problem.d)
            best_coh = coherence(best_frame)

            is_optimal = False

            for _ in range(1, self.max_gen + 1):
                gen_best_frame, _ = solver.step()
                gen_best_coh = coherence(gen_best_frame)

                if gen_best_coh < best_coh:
                    best_coh = gen_best_coh
                    best_frame = gen_best_frame

                if best_coh < coh_lower_bound or math.isclose(
                    best_coh, coh_lower_bound, abs_tol=1e-9
                ):
                    is_optimal = True
                    break

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
