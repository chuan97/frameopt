from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from math import exp, lgamma, log
from typing import Any

import numpy as np

from evomof.core.energy import diff_coherence
from evomof.core.frame import Frame
from evomof.core.manifold import PRODUCT_CP, Chart

__all__ = ["RiemannianCMAConfig", "RiemannianCMA"]


# ----------------------------- helpers -------------------------------------


def _chi_mean(k: int) -> float:
    """E||N(0, I_k)|| — mean of the chi distribution with k dof (stable).

    Computed via log‑gamma to avoid overflow: √2·Γ((k+1)/2)/Γ(k/2).
    """
    if k <= 0:
        return 0.0
    a = 0.5 * log(2.0) + lgamma((k + 1.0) / 2.0) - lgamma(k / 2.0)

    return exp(a)


# ----------------------------- RCMA core ------------------------------------


@dataclass(frozen=True, slots=True)
class RiemannianCMAConfig:
    """
    Configuration parameters for Riemannian CMA-ES.

    Attributes:
        popsize (int): Population size (λ), number of candidate
            solutions per generation.
        parents (int): Number of parents (μ) used for recombination.
        sigma0 (float): Initial global step-size (standard deviation).
        seed (int): Random seed for reproducibility.
        jitter (float): Small regularization term to ensure numerical stability.
        transport_eigs (bool): Whether to transport eigenvectors to new mean frame.
    """

    popsize: int = 32  # λ
    parents: int = 16  # μ
    sigma0: float = 0.5
    seed: int = 0
    jitter: float = 1e-12
    transport_eigs: bool = True  # transport eigenvectors to new mean


@dataclass(slots=True)
class _State:
    X: Frame
    sigma: float
    B: np.ndarray  # (k×k) orthonormal
    D: np.ndarray  # (k,) nonnegative
    p_sigma: np.ndarray  # (k,)
    p_c: np.ndarray  # (k,)
    weights: np.ndarray  # (λ,)
    mu_eff: float
    c_sigma: float
    d_sigma: float
    c_c: float
    c1: float
    c_mu: float


class RiemannianCMA:
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
        if popsize is None:
            dim = max(1, 2 * n * (d - 1))
            popsize = 4 + int(3 * np.log(dim))

        if seed is None:
            # Draw a fresh 32‑bit seed from OS entropy for reproducibility when logged
            seed = int(np.random.SeedSequence().generate_state(1)[0])

        parents = max(1, popsize // 2)
        self.cfg = RiemannianCMAConfig(
            popsize=popsize, parents=parents, sigma0=sigma0, seed=seed
        )
        self.rng = np.random.default_rng(seed)

        if start_frame is not None:
            if start_frame.shape != (n, d):
                raise ValueError(
                    "start_frame dimensions mismatch: "
                    f"expected ({n},{d}), got ({start_frame.shape})"
                )
            start_frame = start_frame.copy()
        else:
            start_frame = Frame.random(n, d, rng=self.rng)

        self.energy_fn = partial(energy_fn, **(energy_kwargs or {}))
        self.state = self._init_state(start_frame)
        self._E_norm = _chi_mean(self.k)
        self._last_steps: list[np.ndarray] = []
        self._iter: int = 0

    @property
    def k(self) -> int:
        n, d = self.state.X.shape

        return 2 * n * (d - 1)

    @staticmethod
    def _weights(lam: int, mu: int) -> tuple[np.ndarray, float, np.ndarray]:
        # Ensure 1 ≤ mu ≤ lam to avoid negative padding or empty weights
        mu = int(min(max(1, mu), lam))

        # Base ranking weights for all ranks 1..λ (Hansen-style)
        idx = np.arange(1, lam + 1)
        base = np.log((lam + 1.0) / 2.0) - np.log(idx)

        # Split and normalize
        w_pos_raw = base[:mu]
        w_pos = w_pos_raw / np.sum(w_pos_raw)
        mu_eff = 1.0 / np.sum(w_pos**2)

        w_neg_raw = base[mu:]
        if w_neg_raw.size > 0:
            w_neg_abs = np.abs(w_neg_raw)
            w_neg_abs = w_neg_abs / np.sum(w_neg_abs)  # ℓ1-normalized positives
        else:
            w_neg_abs = np.zeros(0, dtype=float)

        return w_pos.astype(float), float(mu_eff), w_neg_abs.astype(float)

    def _init_state(self, X0: Frame) -> _State:
        n, d = X0.shape
        k = 2 * n * (d - 1)
        lam = self.cfg.popsize
        mu = self.cfg.parents

        # Stage 1: base weights (positives + unscaled negative template) to get mu_eff
        w_pos, mu_eff, w_neg_abs = self._weights(lam, mu)

        # --- dimension-dependent CMA defaults (Julia-aligned, always automatic) ---
        n_dim = float(k)
        mu_eff_ = float(mu_eff)

        # step-size parameters
        c_sigma = (mu_eff_ + 2.0) / (n_dim + mu_eff_ + 5.0)
        d_sigma = (
            1.0
            + c_sigma
            + 2.0 * max(0.0, np.sqrt((mu_eff_ - 1.0) / (n_dim + 1.0)) - 1.0)
        )

        # covariance path parameter
        c_c = (4.0 + mu_eff_ / n_dim) / (n_dim + 4.0 + 2.0 * mu_eff_ / n_dim)

        # rank-1 and rank-mu learning rates (Julia-aligned)
        c1 = 2.0 / (((n_dim + 1.3) ** 2) + mu_eff_)
        alpha_mu = 2.0
        num = 0.25 + mu_eff_ + 1.0 / mu_eff_ - 2.0
        den = ((n_dim + 2.0) ** 2) + alpha_mu * mu_eff_ / 2.0
        c_mu = min(1.0 - c1, alpha_mu * num / den)

        # Stage 2: always-on active CMA — scale negatives per Hansen/Julia safeguards
        kdim = k
        if w_neg_abs.size > 0:
            # negative scaling candidates
            s1 = 1.0 + c1 / max(c_mu, 1e-16)
            s2 = 1.0 + 2.0 * mu_eff / (kdim + 2.0)
            s3 = max(
                0.0,
                (1.0 - c1 - c_mu) / max(kdim * c_mu, 1e-16),
            )
            scale_neg = -min(s1, s2, s3)
            weights = np.concatenate([w_pos, scale_neg * w_neg_abs])
        else:
            weights = w_pos

        B = np.eye(k)
        D = np.ones(k)
        p_sigma = np.zeros(k)
        p_c = np.zeros(k)

        return _State(
            X=X0,
            sigma=self.cfg.sigma0,
            B=B,
            D=D,
            p_sigma=p_sigma,
            p_c=p_c,
            weights=weights,
            mu_eff=mu_eff,
            c_sigma=float(c_sigma),
            d_sigma=float(d_sigma),
            c_c=float(c_c),
            c1=float(c1),
            c_mu=float(c_mu),
        )

    # --------------------------- ask / tell ---------------------------------

    def ask(self) -> list[Frame]:
        """
        Generate a population of candidate frames by sampling from
        the current search distribution.

        Returns:
            list[Frame]: list of candidate frames sampled from the current
                mean and covariance.
        """
        st = self.state
        lam = self.cfg.popsize
        k = self.k
        z = self.rng.standard_normal((k, lam))
        y = st.B @ (st.D[:, None] * z)  # k×λ
        chart = Chart.at(st.X)
        cands: list[Frame] = []
        self._last_steps = []
        for i in range(lam):
            U = chart.decode(y[:, i])
            Xi = PRODUCT_CP.retract(st.X, st.sigma * U)
            cands.append(Xi)
            self._last_steps.append(y[:, i].copy())
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
        self._iter += 1
        st, cfg = self.state, self.cfg
        lam = len(candidates)
        order = np.argsort(energies)
        mu = min(cfg.parents, lam)
        sel = order[:mu]

        # Recombination in coords (old chart)
        W = st.weights[:lam]
        w_pos = W[:mu]
        Y_sel = np.stack([self._last_steps[i] for i in sel], axis=1)  # k×μ
        y_bar = Y_sel @ w_pos

        # --- all updates in the OLD chart ---------------------------------
        # Step-size path (whitened, old chart)
        By = st.B.T @ y_bar
        y_white = st.B @ (By / np.maximum(st.D, cfg.jitter))  # ~ N(0,I)
        cσ, dσ = st.c_sigma, st.d_sigma
        p_sigma_old = (1.0 - cσ) * st.p_sigma + np.sqrt(
            cσ * (2.0 - cσ) * st.mu_eff
        ) * y_white
        sigma_new = st.sigma * np.exp(
            (cσ / dσ) * (np.linalg.norm(p_sigma_old) / self._E_norm - 1.0)
        )

        # h_sigma gate (Hansen criterion)
        kdim = self.k
        chi = self._E_norm
        # denominator guard to avoid tiny values at early iters
        denom = max(1e-12, np.sqrt(1.0 - (1.0 - cσ) ** (2 * self._iter)))
        hsig = (
            1.0
            if (
                np.linalg.norm(p_sigma_old) / (chi * denom) < (1.4 + 2.0 / (kdim + 1.0))
            )
            else 0.0
        )

        # Covariance path (old chart)
        cc = st.c_c
        p_c_old = (1.0 - cc) * st.p_c + hsig * np.sqrt(
            cc * (2.0 - cc) * st.mu_eff
        ) * y_bar

        # Covariance update (old chart)
        C_old = st.B @ np.diag(st.D**2) @ st.B.T
        c1, cμ = st.c1, st.c_mu
        # Base scaling includes delta_hσ and the (positive+negative) weight sum
        W = st.weights[:lam]
        sum_w = float(np.sum(W))
        delta_h = (1.0 - hsig) * cc * (2.0 - cc)
        C_old *= 1.0 + c1 * delta_h - c1 - cμ * sum_w
        C_old += c1 * np.outer(p_c_old, p_c_old)

        # Positive weights
        for j in range(mu):
            yj = Y_sel[:, j]
            C_old += cμ * w_pos[j] * np.outer(yj, yj)

        # Active (negative) weights — PSD-safe scaling (always on)
        if lam > mu:
            neg_idx = order[mu:]
            Y_neg = np.stack([self._last_steps[i] for i in neg_idx], axis=1)  # k×(λ-μ)
            w_neg = W[mu:]
            if Y_neg.size > 0:
                # Whitened scaling to bound downdates and keep C PSD
                Yw = st.B.T @ Y_neg
                Yw = Yw / np.maximum(st.D[:, None], cfg.jitter)
                scales = kdim / np.maximum(np.sum(Yw**2, axis=0), cfg.jitter)
                for j in range(Y_neg.shape[1]):
                    C_old += (
                        cμ * w_neg[j] * scales[j] * np.outer(Y_neg[:, j], Y_neg[:, j])
                    )

        # --- move the mean and build the new chart -------------------------
        chart_X = Chart.at(st.X)
        U_bar = chart_X.decode(y_bar)
        X_new = PRODUCT_CP.retract(st.X, st.sigma * U_bar)
        chart_Y = Chart.at(X_new)

        # Build T (old → new) by transporting the standard basis in coordinates
        k = self.k
        Id = np.eye(k, dtype=np.float64)
        T_cols = [chart_X.transport_coords(chart_Y, Id[:, j]) for j in range(k)]
        T = np.column_stack(T_cols)

        # --- push covariance and paths into the NEW chart ------------------
        C = T @ C_old @ T.T
        C += cfg.jitter * np.eye(k)
        evals, evecs = np.linalg.eigh(C)
        evals = np.maximum(evals, cfg.jitter)
        B_new = evecs
        D_new = np.sqrt(evals)

        p_sigma_new = T @ p_sigma_old
        p_c_new = T @ p_c_old

        # Commit new state
        self.state = _State(
            X=X_new,
            sigma=sigma_new,
            B=B_new,
            D=D_new,
            p_sigma=p_sigma_new,
            p_c=p_c_new,
            weights=st.weights,
            mu_eff=st.mu_eff,
            c_sigma=st.c_sigma,
            d_sigma=st.d_sigma,
            c_c=st.c_c,
            c1=st.c1,
            c_mu=st.c_mu,
        )
        self._last_steps = []

    # ------------------------------ driver ----------------------------------

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
        i_min = int(np.argmin(energies))
        best_X = cands[i_min]
        best_energy = energies[i_min]
        self.tell(cands, energies)
        return best_X, best_energy

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
