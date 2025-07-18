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

    Attributes
    ----------
    sigma : float
        Current step-size of the optimizer.
    """

    def __init__(
        self,
        n: int,
        d: int,
        sigma0: float = 0.2,
        popsize: int | None = None,
        seed: int | None = None,
        *,
        energy_fn: Callable[..., float] = diff_coherence,
        energy_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        energy_kwargs :
            Extra keyword arguments forwarded to *energy_fn* via
            :pyfunc:`functools.partial`.

        Parameters
        ----------
        seed :
            Optional integer seed for random number generation.
        """
        self.n, self.d = n, d

        rng_gen = np.random.default_rng(seed)
        cma_seed = seed
        cma_opts: dict[str, Any] = {}
        if popsize is not None:
            cma_opts["popsize"] = popsize
        if cma_seed is not None:
            cma_opts["seed"] = cma_seed

        # Final energy callable expects exactly one Frame positional arg.
        self.energy_fn: Callable[[Frame], float] = partial(
            energy_fn, **(energy_kwargs or {})
        )

        mean = Frame.random(n, d, rng=rng_gen)
        self._es = cmaes.CMAEvolutionStrategy(
            frame_to_realvec(mean),
            sigma0,
            cma_opts,
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

        Raises
        ------
        ValueError
            If `max_gen` is negative.

        Returns
        -------
        Frame
            The best frame found over the entire run, according to
            ``self.energy_fn``.
        """
        if max_gen < 0:
            raise ValueError(f"max_gen must be non-negative, got {max_gen}")
        t0 = time.time()
        # Initialize best_frame randomly to handle max_gen=0 cleanly
        best_frame: Frame = Frame.random(self.n, self.d)
        best_E: float = self.energy_fn(best_frame)

        for g in range(1, max_gen + 1):
            cand, E = self.step()
            if E < best_E:
                best_frame, best_E = cand.copy(), E
            if log_every and g % log_every == 0:
                print(f"gen {g:4d}   energy {E:12.6e}   best {best_E:12.6e}")
        print(f"Finished {max_gen} gens in {time.time()-t0:.1f}s → best {best_E:.6e}")
        return best_frame

    @property
    def sigma(self) -> float:
        """
        Get the current step-size (σ) of the underlying CMA-ES optimizer.

        Returns
        -------
        float
            The current CMA-ES step-size (sigma).
        """
        return float(self._es.sigma)
