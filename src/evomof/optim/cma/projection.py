from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np

import cma.evolution_strategy as cmaes
from evomof.core.energy import diff_coherence
from evomof.core.frame import Frame

from .utils import frame_to_realvec, realvec_to_frame


class ProjectionCMA:
    """
    Projection-based CMA-ES for unit-norm frames.

    1. CMA samples in the ambient R^{2nd}.
    2. Each sample is reshaped → Frame → renormalised (projection).
    3. Energy is evaluated on the projected frame.
    """

    def __init__(
        self,
        n: int,
        d: int,
        sigma0: float = 0.2,
        popsize: int | None = None,
        rng: int | np.random.Generator | None = None,
        *,
        energy_fn: Callable[..., float] = diff_coherence,
        energy_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        energy_kwargs :
            Extra keyword arguments forwarded to *energy_fn* via
            :pyfunc:`functools.partial`.
        """
        self.n, self.d = n, d

        # Convert int seed to Generator for mypy compatibility
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        rng_gen: np.random.Generator | None = rng

        # Final energy callable expects exactly one Frame positional arg.
        self.energy_fn: Callable[[Frame], float] = partial(
            energy_fn, **(energy_kwargs or {})
        )

        mean = Frame.random(n, d, rng=rng_gen)
        self._es = cmaes.CMAEvolutionStrategy(
            frame_to_realvec(mean),
            sigma0,
            {"popsize": popsize} if popsize else {},
        )

    def step(self) -> tuple[Frame, float]:
        """
        Execute **one generation** of projection‑CMA‑ES.

        Workflow
        --------
        1. Ask pycma for a batch of candidate vectors (ambient ℝ).
        2. Reshape each into a :class:`Frame`, then project back onto the
           unit‑norm manifold via :pymeth:`Frame.renormalise`.
        3. Evaluate the energy function on each projected frame.
        4. Tell pycma the fitness values to update its internal state.
        5. Return the best projected frame of this generation and its energy.

        Returns
        -------
        tuple[Frame, float]
            *Best* frame (after projection) and its energy in the current
            generation.
        """
        ask = self._es.ask()
        frames = [realvec_to_frame(x, self.n, self.d) for x in ask]
        for fr in frames:
            fr.renormalise()  # project
        energies = [self.energy_fn(fr) for fr in frames]
        self._es.tell(ask, energies)
        best_idx = int(np.argmin(energies))
        return frames[best_idx], float(energies[best_idx])

    def ask(self) -> list[Frame]:
        """
        Sample a new population from CMA-ES and project each into a Frame.

        Returns
        -------
        list[Frame]
            List of projected Frame objects sampled from CMA.
        """
        raw = self._es.ask()
        frames = [realvec_to_frame(x, self.n, self.d) for x in raw]
        return frames

    def tell(self, frames: Sequence[Frame], energies: Sequence[float]) -> None:
        """
        Reinject evaluated frames and their energies into the CMA-ES optimizer.

        Parameters
        ----------
        frames
            Sequence of Frame objects whose fitness has been evaluated.
        energies
            Sequence of objective values corresponding to each frame.
        """
        asks = [frame_to_realvec(fr) for fr in frames]
        # Tell the CMA-ES instance the evaluated fitness values
        self._es.tell(asks, list(energies))

    def run(self, max_gen: int = 200, log_every: int = 10) -> Frame:
        """
        Run the optimiser for a fixed number of generations.

        Parameters
        ----------
        max_gen :
            Total number of generations to execute.
        log_every :
            Print progress every *log_every* generations.  Set to ``0`` to
            disable console output.

        Returns
        -------
        Frame
            The best frame found over the entire run, according to
            ``self.energy_fn``.
        """
        t0 = time.time()
        # First generation establishes a concrete best_frame (non‑None).
        cand, E = self.step()
        best_frame: Frame = cand.copy()
        best_E: float = E

        for g in range(2, max_gen + 1):
            cand, E = self.step()
            if E < best_E:
                best_frame, best_E = cand.copy(), E
            if g % log_every == 0:
                print(f"gen {g:4d}   energy {E:12.6e}   best {best_E:12.6e}")
        print(f"Finished {max_gen} gens in {time.time()-t0:.1f}s → best {best_E:.6e}")
        return best_frame
