from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import cma.evolution_strategy as cmaes
import numpy as np

from frameopt.core._types import Float64Array
from frameopt.core.energy import diff_coherence
from frameopt.core.frame import Frame

from .utils import frame_to_realvec, realvec_to_frame


class ProjectionCMA:
    """
    Projection-based CMA-ES for unit-norm frames.

    1. CMA samples in the ambient R^{2nd}.
    2. Each sample is reshaped → Frame → normalized (projection).
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
        start_frame: Frame | None = None,
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
        start_frame :
            Optional initial Frame whose flattened vector is used as the CMA
            mean.  If *None* (default), a random frame is generated.
        """
        self.n, self.d = n, d

        rng_gen = np.random.default_rng(seed)

        cma_seed = seed
        cma_opts: dict[str, Any] = {}

        if popsize is not None:
            cma_opts["popsize"] = popsize
        else:
            dim = 2 * n * d
            cma_opts["popsize"] = 4 + int(3 * np.log(dim))

        if cma_seed is not None:
            cma_opts["seed"] = cma_seed

        cma_opts["verb_disp"] = 0

        # Final energy callable expects exactly one Frame positional arg.
        self.energy_fn: Callable[[Frame], float] = partial(
            energy_fn, **(energy_kwargs or {})
        )

        if start_frame is not None:
            if start_frame.shape != (n, d):
                raise ValueError(
                    "start_frame dimensions mismatch: "
                    f"expected ({n},{d}), got ({start_frame.shape})"
                )
            mean_frame = start_frame.copy()
        else:
            mean_frame = Frame.random(n, d, rng=rng_gen)

        self._es = cmaes.CMAEvolutionStrategy(
            frame_to_realvec(mean_frame),
            sigma0,
            cma_opts,
        )

    @property
    def sigma(self) -> float:
        """
        Current step-size (σ) of the underlying CMA-ES optimizer.
        """
        return float(self._es.sigma)

    @property
    def mean(self) -> np.ndarray:
        """Current mean vector of the CMA-ES search distribution."""
        arr: np.ndarray = np.asarray(self._es.mean)  # typed boundary: normalize type

        return arr

    def step(self) -> tuple[Frame, float]:
        """
        Execute **one generation** of projection‑CMA‑ES.

        Workflow
        --------
        1. Ask pycma for a batch of candidate vectors (ambient ℝ).
        2. Reshape each into a :class:`Frame` and, if needed, project/normalize
           onto the unit‑norm manifold.
        3. Evaluate the energy function on each projected frame.
        4. Tell pycma the fitness values to update its internal state.
        5. Return the best projected frame of this generation and its energy.

        Returns
        -------
        tuple[Frame, float]
            *Best* frame (after projection) and its energy in the current
            generation.
        """
        raws = self.ask()
        frames = [realvec_to_frame(x, self.n, self.d) for x in raws]
        energies = [self.energy_fn(fr) for fr in frames]
        self.tell(raws, energies)
        best_idx = np.argmin(energies)

        return frames[best_idx], energies[best_idx]

    def ask(self) -> list[Float64Array]:
        """
        Sample a new population of **raw ambient vectors** from CMA‑ES.

        Returns
        -------
        list[Float64Array]
            The raw vectors returned by :meth:`CMAEvolutionStrategy.ask`.
        """
        raws: list[Float64Array] = self._es.ask()

        return raws

    def tell(self, raws: Sequence[Float64Array], energies: Sequence[float]) -> None:
        """
        Report fitness values for the **raw vectors** produced by :meth:`ask`.

        Parameters
        ----------
        raws
            Sequence of raw ambient vectors (same order/batch as returned by
            :meth:`ask` in the current generation).
        energies
            Objective values corresponding to each vector in ``raws``.
        """
        self._es.tell(raws, energies)

    def run(self, max_gen: int = 200, tol: float = 1e-12, log_every: int = 10) -> Frame:
        """
        Run the optimiser until convergence or a generation cap is reached.

        Run until the best energy **of the current generation** changes by less than
        `tol`, or until `max_gen` generations have been executed.

        Parameters
        ----------
        max_gen :
            Total number of generations to execute.
        tol :
            Absolute tolerance on the improvement of the best energy.
            Optimisation stops when ``abs(E_best_prev - E_best) < tol``.
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
        prev_E: float | None = None  # track energy of previous generation

        for g in range(1, max_gen + 1):
            cand, E = self.step()

            if E < best_E:
                best_frame, best_E = cand.copy(), E

            if log_every and g % log_every == 0:
                print(f"gen {g:4d}   energy {E:12.6e}   best {best_E:12.6e}")

            # convergence check on generation‑best energy
            if tol > 0 and prev_E is not None and abs(prev_E - E) < tol:
                if log_every:
                    print(f"Converged (|ΔE| < {tol}) at generation {g}")

                break

            prev_E = E

        if log_every:
            print(f"Finished {g} gens in {time.time()-t0:.1f}s → best {best_E:.6e}\n")

        return best_frame
