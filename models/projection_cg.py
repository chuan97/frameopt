from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import yaml
from projection_pramp import ProjectionPRampModel

from frameopt.core.energy import coherence, diff_coherence, grad_diff_coherence
from frameopt.core.frame import Frame
from frameopt.model.api import Problem, Result
from frameopt.optim.local import cg_minimize


@dataclass(frozen=True, slots=True)
class ProjectionCGModel:
    pmodel: ProjectionPRampModel
    maxiter: int = 100
    p: int = 100

    @classmethod
    def from_config(cls, path: Path) -> ProjectionCGModel:
        cfg = yaml.safe_load(path.read_text())
        init = cfg["init"]

        pmodelcfg_path = init.pop("pmodel")
        pmodel = ProjectionPRampModel.from_config(Path(pmodelcfg_path))
        init["pmodel"] = pmodel

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

        t0 = time.perf_counter()
        result = self.pmodel.run(problem)
        best_frame = cg_minimize(
            frame0=result.best_frame,
            energy_fn=partial(diff_coherence, p=self.p),
            grad_fn=partial(grad_diff_coherence, p=self.p),
            maxiter=self.maxiter,
        )
        dt = time.perf_counter() - t0

        return Result(
            problem=problem,
            best_frame=best_frame,
            best_coherence=coherence(best_frame),
            wall_time_s=dt,
        )
