"""
CMA-ES optimizer with Riemannian manifold support.

Adaptation of Manopt.jl implementation of Riemannian CMA-ES

Reference: [Hansen 2023] N. Hansen, The CMA Evolution Strategy: A Tutorial, arXiv:1604.00772 (2023)
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from frameopt.core.energy import diff_coherence
from frameopt.core.frame import Frame
from frameopt.core.manifold import PRODUCT_CP, Chart, ProductCP

__all__ = ["RiemannianCMAParams", "BaseCMAParams", "RiemannianCMABase", "RiemannianCMA"]


def _chi_mean(k: int) -> float:
    """E‖𝒩(0, I_k)‖ — chi mean (asymptotic approximation).

    Uses the standard expansion:
        √k · (1 − 1/(4k) + 1/(21 k²))
    """
    if k <= 0:
        return 0.0
    return math.sqrt(k) * (1.0 - 1.0 / (4.0 * k) + 1.0 / (21.0 * k * k))


@dataclass(frozen=True, slots=True)
class RiemannianCMAParams:
    """
    Configuration parameters for Riemannian CMA-ES.

    Attributes:
    base (BaseCMAParams):
        Base CMA-ES parameters.
        geom (ProductCP): Manifold geometry to use.
    """

    base: BaseCMAParams
    geom: ProductCP = PRODUCT_CP


@dataclass(frozen=True, slots=True)
class BaseCMAParams:
    """
    Run-constant CMA-ES parameters in Euclidean dimension ``k``.

    This is **manifold-agnostic**: it encodes only the base CMA hyperparameters
    (selection weights and learning rates) and generic run constants.

    Attributes
    ----------
    k : int
        Intrinsic search dimension.
    lambda_ : int
        Population size (λ).
    mu : int
        Number of parents (μ) used for recombination.
    weights : numpy.ndarray
        Length-λ vector of selection weights (includes active/negative part if used).
    mu_eff : float
        Effective parent number (from the positive block).
    c_sigma, d_sigma, c_c, c1, c_mu, c_m : float
        Standard CMA learning-rate hyperparameters.
    chi_mean : float
        E‖𝒩(0, I_k)‖ used by step-size control.
    sigma0 : float
        Initial global step-size.
    jitter : float
        Numerical floor used in covariance regularization.
    seed : int
        RNG seed used to initialize the generator.
    """

    k: int
    lambda_: int
    mu: int
    weights: np.ndarray
    mu_eff: float
    c_sigma: float
    d_sigma: float
    c_c: float
    c1: float
    c_mu: float
    c_m: float
    chi_mean: float
    sigma0: float
    jitter: float
    seed: int

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be positive.")
        if self.lambda_ < 2:
            raise ValueError("lambda_ must be at least two.")
        if not (1 <= self.mu <= self.lambda_):
            raise ValueError("mu must satisfy 1 ≤ mu ≤ lambda_.")
        if not 1 <= self.mu_eff <= self.mu:
            raise ValueError("mu_eff must satisfy 1 ≤ mu_eff ≤ mu.")
        if not isinstance(self.weights, np.ndarray):
            raise TypeError("weights must be a numpy.ndarray.")
        if self.weights.shape != (self.lambda_,):
            raise ValueError("weights must have shape (lambda_,).")
        if self.weights.ndim != 1 or self.weights.size != self.lambda_:
            raise ValueError("weights must be a 1D array of length lambda_.")
        if np.any(self.weights[: self.mu] < 0):
            raise ValueError("First mu weights must be positive.")
        if not np.isclose(np.sum(self.weights[: self.mu]), 1.0):
            raise ValueError("First mu weights must sum to one.")
        if np.any(self.weights[self.mu :] > 0):
            raise ValueError("Last lambda_ - mu weights must be non-positive.")
        if np.any(self.weights[:-1] - self.weights[1:] < 0):
            raise ValueError("Weights must be in non-ascending order.")
        if not 0 < self.c_sigma < 1:
            raise ValueError("c_sigma must be in (0, 1).")
        if self.d_sigma <= 0:
            raise ValueError("d_sigma must be positive.")
        if not 0 <= self.c_m <= 1:
            raise ValueError("c_m must be in [0, 1].")
        if not 0 <= self.c_c <= 1:
            raise ValueError("c_c must be in [0, 1].")
        if not 0 <= self.c1 <= 1 - self.c_mu:
            raise ValueError("c_1 must be in [0, 1 - c_mu]")
        if not 0 <= self.c_mu <= 1 - self.c1:
            raise ValueError("c_mu must be in [0, 1 - c_1]")

    @classmethod
    def auto(
        cls,
        k: int,
        popsize: int | None = None,
        *,
        sigma0: float = 0.5,
        jitter: float = 1.0e-12,
        seed: int = 42,
    ) -> BaseCMAParams:
        if popsize is None:
            lam = 4 + int(3 * np.log(k))
        else:
            lam = popsize
        mu = lam // 2

        # selection and recombination
        idx = np.arange(1, lam + 1)
        base = np.log((lam + 1.0) / 2.0) - np.log(idx)  # Eq. (49) Hansen 2023
        w_pos = base[:mu]
        w_neg = base[mu:]
        mu_eff = np.sum(w_pos) ** 2 / np.sum(w_pos**2)
        mu_eff_neg = np.sum(w_neg) ** 2 / np.sum(w_neg**2)
        w_pos = w_pos / np.sum(w_pos)  # Top term Eq. (53) Hansen 2023

        c_m = 1.0  # Eq. (54) Hansen 2023

        # step-size control
        c_sigma = (mu_eff + 2.0) / (k + mu_eff + 5.0)  # Eq. (55) Hansen 2023
        d_sigma = (
            1.0 + c_sigma + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (k + 1.0)) - 1.0)
        )  # Eq. (55) Hansen 2023

        # covariance matrix adaptation
        c_c = (4.0 + mu_eff / k) / (k + 4.0 + 2.0 * mu_eff / k)  # Eq. (56) Hansen 2023
        c1 = 2.0 / (((k + 1.3) ** 2) + mu_eff)  # Eq. (57) Hansen 2023
        num = 0.25 + mu_eff + 1.0 / mu_eff - 2.0
        den = (k + 2.0) ** 2 + 2.0 * mu_eff / 2.0
        c_mu = min(1.0 - c1, 2.0 * num / den)  # Eq. (58) Hansen 2023

        # finish weights (compute negative weights)
        w_neg = w_neg / np.sum(np.abs(w_neg))
        s1 = 1.0 + c1 / c_mu  # Eq. (50) Hansen 2023
        s2 = 1.0 + 2.0 * mu_eff_neg / (mu_eff + 2.0)  # Eq. (51) Hansen 2023
        s3 = (1.0 - c1 - c_mu) / (k * c_mu)  # Eq. (52) Hansen 2023
        scale_neg = min(s1, s2, s3)
        w_neg *= scale_neg  # Bottom term Eq. (53) Hansen 2023
        weights = np.concatenate([w_pos, w_neg])

        return cls(
            k=k,
            lambda_=lam,
            mu=mu,
            weights=weights,
            mu_eff=mu_eff,
            c_sigma=c_sigma,
            d_sigma=d_sigma,
            c_c=c_c,
            c1=c1,
            c_mu=c_mu,
            c_m=c_m,
            chi_mean=_chi_mean(k),
            sigma0=sigma0,
            jitter=jitter,
            seed=seed,
        )


@dataclass(frozen=True, slots=True)
class RiemannianCMAState:
    X: Frame
    chart: Chart
    sigma: float
    iter: int
    B: np.ndarray  # (k×k) orthonormal
    D: np.ndarray  # (k,) nonnegative
    p_sigma: np.ndarray  # (k,)
    p_c: np.ndarray  # (k,)
    last_steps: np.ndarray


class RiemannianCMABase:
    """
    Riemannian CMA-ES optimizer for frames on the unit-norm manifold.

    This class implements a Riemannian variant of the CMA-ES algorithm by working
    in an explicit coordinate system in the tangent space at the current mean frame.
    The algorithm performs CMA updates in Euclidean coordinates and maps between the
    tangent space and the manifold using Frame.retract and Frame.transport methods.

    Usage:
        - Initialize with problem dimensions and optional parameters.
        - Use `ask()` to generate candidate frames.
        - Evaluate candidates and pass energies to `tell()`.
        - Use `step()` to perform a full generation cycle (ask, evaluate, tell).
        - `run()` performs iterative optimization until convergence or max generations.

    Methods:
        ask(): Generate a population of candidate frames.
        tell(candidates, energies): Update the internal state based on
            evaluated energies.
        step(): Perform one generation step, returning best candidate and energy.
        run(): Run the optimizer until convergence or generation limit.
    """

    def __init__(
        self,
        cfg: RiemannianCMAParams,
        start_frame: Frame,
        *,
        energy_fn: Callable[[Frame], float] = diff_coherence,
        energy_kwargs: dict[str, Any] | None = None,
    ):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.base.seed)
        self.energy_fn = partial(energy_fn, **(energy_kwargs or {}))
        self.state = RiemannianCMAState(
            X=start_frame.copy(),
            chart=Chart.at(start_frame, geom=self.geom),
            sigma=cfg.base.sigma0,
            iter=0,
            B=np.eye(self.k),
            D=np.ones(self.k),
            p_sigma=np.zeros(self.k),
            p_c=np.zeros(self.k),
            last_steps=np.empty((self.k, 0), dtype=float),
        )

    @property
    def k(self) -> int:
        """Intrinsic dimension of the search space (tangent space at mean)."""
        return self.cfg.base.k

    @property
    def geom(self) -> ProductCP:
        """Manifold geometry used by the optimizer."""
        return self.cfg.geom

    @property
    def mean(self) -> Frame:
        """Current mean :class:`Frame`."""
        return self.state.X

    @property
    def sigma(self) -> float:
        """Current global step‑size σ."""
        return self.state.sigma

    def ask(self) -> list[Frame]:
        """
        Generate a population of candidate frames by sampling from
        the current search distribution.

        Returns:
            list[Frame]: list of candidate frames sampled from the current
                mean and covariance.
        """
        st = self.state
        lam = self.cfg.base.lambda_

        # sample new population (in old chart)
        z = self.rng.standard_normal((self.k, lam))  # Eq. (38) Hansen 2023
        y = st.B @ (st.D[:, None] * z)  # Eq. (39) Hansen 2023

        cands: list[Frame] = []
        for i in range(lam):
            U = st.chart.decode(
                y[:, i]
            )  # decode each candidate difference into a tangent vector
            Xi = self.geom.retract(
                st.X, st.sigma * U
            )  # Eq. (40) Hansen 2023 (retract the tangent vector to get the candidate frames)

            cands.append(Xi)

        self.state = RiemannianCMAState(
            X=st.X,
            chart=st.chart,
            sigma=st.sigma,
            iter=st.iter,
            B=st.B,
            D=st.D,
            p_sigma=st.p_sigma,
            p_c=st.p_c,
            last_steps=y,
        )
        return cands

    def tell(self, candidates: list[Frame], energies: np.ndarray) -> None:
        """
        Update the internal state of the optimizer based on evaluated
        energies of candidates.

        Args:
            candidates (list[Frame]): list of candidate frames generated by `ask()`.
            energies (np.ndarray): Array of energy values corresponding to
                each candidate.
        """
        st = self.state
        base = self.cfg.base
        geom = self.geom
        k = self.k
        mu = base.mu
        c_sigma = base.c_sigma
        mu_eff = base.mu_eff
        chi_mean = base.chi_mean
        cc = base.c_c
        c1, c_mu = base.c1, base.c_mu
        w = base.weights.copy()
        iter_new = st.iter + 1

        # selection and recombination (in old chart)
        order = np.argsort(energies)
        sel = order[:mu]
        w_pos = w[:mu]
        y_sel = st.last_steps[:, sel]  # k×μ
        y_bar = y_sel @ w_pos  # Eq. (41) Hansen 2023
        chart_X = st.chart
        U_bar = chart_X.decode(y_bar)
        X_new = geom.retract(st.X, base.c_m * st.sigma * U_bar)  # Eq. (42) Hansen 2023
        chart_Y = chart_X.transport_to(X_new)

        # step-size control (old chart)
        By = st.B.T @ y_bar
        y_white = st.B @ (By / st.D)  # ~ N(0,I)
        p_sigma_old = (1.0 - c_sigma) * st.p_sigma + np.sqrt(
            c_sigma * (2.0 - c_sigma) * mu_eff
        ) * y_white  # Eq. (43) Hansen 2023
        sigma_new = st.sigma * np.exp(
            (c_sigma / base.d_sigma) * (np.linalg.norm(p_sigma_old) / chi_mean - 1.0)
        )  # Eq. (44) Hansen 2023

        # covariance matrix adaptation (old chart)
        denom = np.sqrt(1.0 - (1.0 - c_sigma) ** (2 * iter_new))
        hsig = (
            1.0
            if (
                np.linalg.norm(p_sigma_old) / (chi_mean * denom)
                < (1.4 + 2.0 / (k + 1.0))
            )
            else 0.0
        )
        p_c_old = (1.0 - cc) * st.p_c + hsig * np.sqrt(
            cc * (2.0 - cc) * mu_eff
        ) * y_bar  # Eq. (45) Hansen 2023
        C_old = st.B @ np.diag(st.D**2) @ st.B.T
        sum_w = np.sum(w)
        delta_h = (1.0 - hsig) * cc * (2.0 - cc)
        C_old *= (
            1.0 + c1 * delta_h - c1 - c_mu * sum_w
        )  # First term Eq. (46) Hansen 2023
        C_old += c1 * np.outer(p_c_old, p_c_old)  # Second term Eq. (46) Hansen 2023
        y_all = st.last_steps[:, order]
        neg_mask = w < 0.0
        y_neg = y_all[:, neg_mask]
        Yw = st.B.T @ y_neg
        Yw = Yw / st.D[:, None]
        scales = k / np.sum(Yw**2, axis=0)
        w[neg_mask] *= scales
        C_old += (
            c_mu * (y_all * w[None, :]) @ y_all.T
        )  # Third term Eq. (46) Hansen 2023

        # transport covariance eigenbasis and paths to the new mean
        vals, vecs = np.linalg.eigh(C_old)
        D_new = np.sqrt(np.maximum(vals, base.jitter))
        B_new = chart_X.transport_basis(chart_Y, vecs)
        p_sigma_new = chart_X.transport_coords(chart_Y, p_sigma_old)
        p_c_new = chart_X.transport_coords(chart_Y, p_c_old)

        self.state = RiemannianCMAState(
            X=X_new,
            chart=chart_Y,
            sigma=float(sigma_new),
            iter=int(iter_new),
            B=B_new,
            D=D_new,
            p_sigma=p_sigma_new,
            p_c=p_c_new,
            last_steps=st.last_steps,
        )

    def step(self) -> tuple[Frame, float]:
        """
        Perform one iteration (generation) of the optimizer.

        This method performs the full ask-evaluate-tell cycle:
        - Generates candidate frames.
        - Evaluates their energies using the configured energy function.
        - Updates the internal state based on the evaluated energies.

        Returns:
            tuple[Frame, float]: The best candidate frame and its energy value from
                this generation.
        """
        cands = self.ask()
        energies = np.array([self.energy_fn(Xi) for Xi in cands], dtype=float)
        best_idx = np.argmin(energies)
        self.tell(cands, energies)

        return cands[best_idx], energies[best_idx]

    def run(
        self,
        max_gen: int = 200,
        tol: float = 1e-12,
        log_every: int = 10,
    ) -> Frame:
        """
        Run the optimizer until convergence or a maximum number
        of generations is reached.

        The optimization stops when the change in the best energy
        of the current generation falls below the specified tolerance `tol`,
        or when `max_gen` generations have been run.

        Args:
            max_gen (int): Maximum number of generations to run (non-negative).
            tol (float): Convergence tolerance for change in best energy.
            log_every (int): Frequency of logging progress (in generations).

        Returns:
            Frame: The best frame found after optimization.
        """
        if max_gen < 0:
            raise ValueError(f"max_gen must be non-negative, got {max_gen}")

        t0 = time.time()
        # Initialize best_frame randomly to handle max_gen=0 cleanly
        n, d = self.state.X.shape
        best_frame: Frame = Frame.random(n, d)
        best_E: float = float(self.energy_fn(best_frame))
        prev_E: float | None = None  # track energy of previous generation

        for g in range(1, max_gen + 1):
            cand, E = self.step()

            if E < best_E:
                best_frame, best_E = cand.copy(), float(E)

            if log_every and g % log_every == 0:
                print(f"gen {g:4d}   energy {E:12.6e}   best {best_E:12.6e}")

            # convergence check on generation‑best energy
            if tol > 0 and prev_E is not None and abs(prev_E - float(E)) < tol:
                if log_every:
                    print(f"Converged (|ΔE| < {tol}) at generation {g}")

                break

            prev_E = float(E)

        print(f"Finished {g} gens in {time.time()-t0:.1f}s → best {best_E:.6e}\n")

        return best_frame


class RiemannianCMA:
    """
    Convenience wrapper around :class:`RiemannianCMABase` with a user-friendly
    constructor mirroring ProjectionCMA.
    """

    def __init__(
        self,
        n: int,
        d: int,
        sigma0: float = 0.3,
        start_frame: Frame | None = None,
        popsize: int | None = None,
        seed: int | None = None,
        *,
        energy_fn: Callable[[Frame], float] = diff_coherence,
        energy_kwargs: dict[str, Any] | None = None,
    ):
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])

        if start_frame is not None:
            if start_frame.shape != (n, d):
                raise ValueError(
                    "start_frame dimensions mismatch: "
                    f"expected ({n},{d}), got ({start_frame.shape})"
                )
            X0 = start_frame
        else:
            X0 = Frame.random(n, d, rng=np.random.default_rng(seed))

        k = 2 * n * (d - 1)
        base = BaseCMAParams.auto(k=k, popsize=popsize, sigma0=sigma0, seed=seed)
        geom = ProductCP(
            retraction_kind="exponential",
            transport_kind="projection",
        )
        params = RiemannianCMAParams(base=base, geom=geom)

        self.impl = RiemannianCMABase(
            params, X0, energy_fn=energy_fn, energy_kwargs=energy_kwargs
        )

    @property
    def k(self) -> int:
        return self.impl.k

    @property
    def mean(self) -> Frame:
        return self.impl.mean

    @property
    def sigma(self) -> float:
        return self.impl.sigma

    def ask(self) -> list[Frame]:
        return self.impl.ask()

    def tell(self, candidates: list[Frame], energies: np.ndarray) -> None:
        self.impl.tell(candidates, energies)

    def step(self) -> tuple[Frame, float]:
        return self.impl.step()

    def run(
        self,
        max_gen: int = 200,
        tol: float = 1e-12,
        log_every: int = 10,
    ) -> Frame:
        return self.impl.run(max_gen=max_gen, tol=tol, log_every=log_every)
