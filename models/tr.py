from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from frameopt.core.energy import coherence
from frameopt.core.frame import Frame
from frameopt.model.api import Problem, Result
from frameopt.optim.local import tr_minimize


@dataclass(frozen=True, slots=True)
class TRModel:
    energy_func: Callable[[Frame], float]
    grad_func: Callable[[Frame], Any]
    maxiter: int = 100
    seed: int | None = None

    @classmethod
    def from_config(cls, path: Path) -> TRModel:
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

        rng_gen = np.random.default_rng(self.seed)
        start_frame = Frame.random(n=problem.n, d=problem.d, rng=rng_gen)

        t0 = time.perf_counter()
        best_frame = tr_minimize(
            frame0=start_frame,
            energy_fn=self.energy_func,
            grad_fn=self.grad_func,
            maxiter=self.maxiter,
        )
        dt = time.perf_counter() - t0

        return Result(
            problem=problem,
            best_frame=best_frame,
            best_coherence=coherence(best_frame),
            wall_time_s=dt,
        )
