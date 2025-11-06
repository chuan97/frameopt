from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from frameopt.bounds import max_lower_bound
from frameopt.core.energy import coherence
from frameopt.core.frame import Frame
from frameopt.model.api import Problem, Result
from frameopt.optim.local import cg_minimize


@dataclass(frozen=True, slots=True)
class CGModel:
    energy_func: Callable[[Frame], float]
    grad_func: Callable[[Frame], Any]
    seed: int
    maxiter: int = 100
    restarts: int = 1

    @classmethod
    def from_config(cls, path: Path) -> CGModel:
        cfg = yaml.safe_load(path.read_text())
        init = cfg["init"]

        ecfg = init.pop("energy")
        mod_name, _, func_name = ecfg["import"].partition(":")
        mod = importlib.import_module(mod_name)
        energy_func = getattr(mod, func_name)
        energy_func = partial(energy_func, **ecfg["kwargs"])

        init["energy_func"] = energy_func

        gcfg = init.pop("grad")
        mod_name, _, func_name = gcfg["import"].partition(":")
        mod = importlib.import_module(mod_name)
        grad_func = getattr(mod, func_name)
        grad_func = partial(grad_func, **ecfg["kwargs"])

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
            rng_gen = np.random.default_rng(seed)
            start_frame = Frame.random(n=problem.n, d=problem.d, rng=rng_gen)

            is_optimal = False

            best_frame = cg_minimize(
                frame0=start_frame,
                energy_fn=self.energy_func,
                grad_fn=self.grad_func,
                maxiter=self.maxiter,
            )
            best_coh = coherence(best_frame)

            if best_coh < overall_best_coh:
                overall_best_coh = best_coh
                overall_best_frame = best_frame

            if best_coh < coh_lower_bound or math.isclose(
                best_coh, coh_lower_bound, abs_tol=1e-9
            ):
                overall_best_frame = best_frame
                overall_best_coh = best_coh
                is_optimal = True
                break

            seed += 1
        dt = time.perf_counter() - t0

        return Result(
            problem=problem,
            best_frame=best_frame,
            best_coherence=coherence(best_frame),
            wall_time_s=dt,
            extras={"optimal": is_optimal},
        )
