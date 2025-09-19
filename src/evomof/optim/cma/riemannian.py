from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from evomof.core._types import Complex128Array, Float64Array
from evomof.core.energy import diff_coherence
from evomof.core.frame import Frame
from evomof.core.manifold import PRODUCT_CP

__all__ = ["RiemannianCMAConfig", "RiemannianCMA"]


# ----------------------------- helpers -------------------------------------


def _chi_mean(k: int) -> float:
    """E||N(0, I_k)|| — mean of the chi distribution with k dof (stable).

    Computed via log‑gamma to avoid overflow: √2·Γ((k+1)/2)/Γ(k/2).
    """
    from math import exp, lgamma, log

    if k <= 0:
        return 0.0
    a = 0.5 * log(2.0) + lgamma((k + 1.0) / 2.0) - lgamma(k / 2.0)
    return float(exp(a))


def _householder_basis(x: np.ndarray) -> Complex128Array:
    """Return Q ∈ C^{d×(d-1)} with columns orthonormal and orthogonal to x.

    x is assumed nonzero; rows of Frame are unit in practice.
    """
    x = np.asarray(x, dtype=np.complex128)
    d = x.shape[0]
    nrm = np.linalg.norm(x)
    if nrm == 0:
        id = np.eye(d, dtype=np.complex128)
        return id[:, 1:]
    v = x / nrm
    a0 = v[0]
    phase = a0 / np.abs(a0) if np.abs(a0) > 0 else 1.0 + 0j
    u = v.copy()
    u[0] += phase
    un = np.linalg.norm(u)
    if un < 1e-14:
        id = np.eye(d, dtype=np.complex128)
        return id[:, 1:]
    u /= un
    H = np.eye(d, dtype=np.complex128) - 2.0 * np.outer(u, np.conj(u))
    Q = H[:, 1:]
    return Q


class _CoordCodec:
    """
    Helper class to encode and decode tangent vectors to/from real coordinate arrays
    at a given Frame.

    Coordinates are real vectors of length k = 2*n*(d-1), obtained by concatenating
    the real and imaginary parts of complex coefficients
    in the per-row Householder basis Q_j.

    Methods:
        encode(U): Convert a tangent vector (complex array) to real coordinates.
        decode(y): Convert real coordinates back to a tangent vector.
        transport_coords(y, X, Y): Transport coordinate vector y from tangent space
            at X to tangent space at Y.
        transport_basis(B, X, Y): Transport a matrix of basis vectors
            from tangent space at X to Y.
    """

    def __init__(self, X: Frame):
        self.X = X
        self.n, self.d = X.shape
        self.m = self.n * (self.d - 1)  # complex coordinate length
        self.k = 2 * self.m  # real coordinate length
        self.Q_blocks: list[Complex128Array] = [
            _householder_basis(X.vectors[i, :]) for i in range(self.n)
        ]

    # tangent (n×d complex) → y ∈ R^k
    def encode(self, U: Complex128Array) -> Float64Array:
        assert U.shape == (self.n, self.d)
        parts: list[Complex128Array] = []
        for i in range(self.n):
            c: Complex128Array = np.conj(self.Q_blocks[i]).T @ U[i, :]
            parts.append(c)
        c_all: Complex128Array = np.concatenate(parts, axis=0)
        y: Float64Array = np.concatenate([c_all.real, c_all.imag], axis=0).astype(
            np.float64, copy=False
        )
        return y

    # y ∈ R^k → tangent (n×d complex)
    def decode(self, y: Float64Array) -> Complex128Array:
        assert y.ndim == 1 and y.shape[0] == self.k
        m = self.m
        c = y[:m] + 1j * y[m:]
        U: Complex128Array = np.empty((self.n, self.d), dtype=np.complex128)
        off = 0
        for i in range(self.n):
            w = c[off : off + (self.d - 1)]
            U[i, :] = self.Q_blocks[i] @ w
            off += self.d - 1
        # safety: enforce tangency against the *current* frame
        U = PRODUCT_CP.project(self.X, U)
        return U

    # transport coord vector y from X to Y
    @staticmethod
    def transport_coords(y: Float64Array, X: Frame, Y: Frame) -> Float64Array:
        codec_X = _CoordCodec(X)
        U = codec_X.decode(y)
        V = PRODUCT_CP.transport(X, Y, U)
        codec_Y = _CoordCodec(Y)
        y_new = codec_Y.encode(V)
        return y_new

    # transport matrix of basis vectors (k×k), column-wise
    @staticmethod
    def transport_basis(B: Float64Array, X: Frame, Y: Frame) -> Float64Array:
        cols = [_CoordCodec.transport_coords(B[:, j], X, Y) for j in range(B.shape[1])]
        B_new = np.stack(cols, axis=1)
        B_new, _ = np.linalg.qr(B_new)
        B_new = B_new.astype(np.float64)
        return B_new


# ----------------------------- RCMA core ------------------------------------


@dataclass
class RiemannianCMAConfig:
    """
    Configuration parameters for Riemannian CMA-ES.

    Attributes:
        popsize (int): Population size (λ), number of candidate
            solutions per generation.
        parents (int): Number of parents (μ) used for recombination.
        sigma0 (float): Initial global step-size (standard deviation).
        c_sigma (float): Learning rate for step-size control.
        d_sigma (float): Damping parameter for step-size update.
        c_c (float): Cumulation parameter for rank-1 update path.
        c1 (float): Learning rate for rank-1 covariance update.
        c_mu (float): Learning rate for rank-μ covariance update.
        use_active (bool): Whether to use active CMA with negative weights.
        seed (int): Random seed for reproducibility.
        jitter (float): Small regularization term to ensure numerical stability.
        transport_eigs (bool): Whether to transport eigenvectors to new mean frame.
        auto_hyper (bool): If True, set CMA learning rates
            (c_sigma, d_sigma, c_c, c1, c_mu) from dimension-dependent defaults at init.
        active_warmup (int): Generations to wait before enabling active CMA.
    """

    popsize: int = 32  # λ
    parents: int = 16  # μ
    sigma0: float = 0.5

    c_sigma: float = 0.3
    d_sigma: float = 1.0

    c_c: float = 0.2  # cumulation for rank-1 path
    c1: float = 2e-2  # learning rate rank-1
    c_mu: float = 3e-2  # learning rate rank-μ

    use_active: bool = False  # negative weights (active CMA)
    seed: int = 0
    jitter: float = 1e-12

    transport_eigs: bool = True  # transport eigenvectors to new mean
    auto_hyper: bool = True
    active_warmup: int = 1000  # generations to wait before enabling active CMA


@dataclass
class _State:
    X: Frame
    sigma: float
    B: np.ndarray  # (k×k) orthonormal
    D: np.ndarray  # (k,) nonnegative
    p_sigma: np.ndarray  # (k,)
    p_c: np.ndarray  # (k,)
    weights: np.ndarray  # (λ,)
    mu_eff: float


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
            popsize = 32
        if seed is None:
            seed = 0
        self.cfg = RiemannianCMAConfig(popsize=popsize, sigma0=sigma0, seed=seed)
        self.rng = np.random.default_rng(seed)

        if start_frame is None:
            # generate default start frame as orthonormal rows (n x d)
            # random complex vectors normalized to 1
            vectors = self.rng.normal(size=(n, d)) + 1j * self.rng.normal(size=(n, d))
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            start_frame = Frame(vectors)
        else:
            if start_frame.shape != (n, d):
                raise ValueError(
                    f"start_frame shape {start_frame.shape} does not match "
                    f"(n,d)=({n},{d})"
                )
        self.energy_fn = partial(energy_fn, **(energy_kwargs or {}))
        self.state = self._init_state(start_frame)
        self._E_norm = _chi_mean(self.k)
        self._last_steps: list[np.ndarray] = []
        self._iter: int = 0

    # dimension helpers
    @property
    def k(self) -> int:
        n, d = self.state.X.shape
        return 2 * n * (d - 1)

    @staticmethod
    def _weights(lam: int, mu: int, use_active: bool) -> tuple[np.ndarray, float]:
        # Ensure 1 ≤ mu ≤ lam to avoid negative padding or empty weights
        mu = int(min(max(1, mu), lam))

        idx = np.arange(1, mu + 1)
        w_pos = np.log(mu + 0.5) - np.log(idx)
        w_pos = w_pos / np.sum(w_pos)
        mu_eff = 1.0 / np.sum(w_pos**2)

        if use_active and lam > mu:
            idx_neg = np.arange(mu + 1, lam + 1)
            w_neg = np.log(mu + 0.5) - np.log(idx_neg)
            denom = np.sum(np.abs(w_neg))
            if denom <= 0:
                w_neg = np.ones_like(w_neg)
                denom = float(w_neg.size)
            w_neg = w_neg / denom
            w = np.concatenate([w_pos, -0.5 * w_neg])
        else:
            # Pad with zeros to length λ if no active part is used
            pad = lam - mu
            if pad > 0:
                w = np.concatenate([w_pos, np.zeros(pad)])
            else:
                w = w_pos

        return w.astype(float), float(mu_eff)

    def _init_state(self, X0: Frame) -> _State:
        n, d = X0.shape
        k = 2 * n * (d - 1)
        lam = self.cfg.popsize
        mu = self.cfg.parents
        weights, mu_eff = self._weights(lam, mu, self.cfg.use_active)
        B = np.eye(k)
        D = np.ones(k)
        p_sigma = np.zeros(k)
        p_c = np.zeros(k)
        # --- dimension-dependent CMA defaults (Hansen) ---
        if self.cfg.auto_hyper:
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

            # rank-1 and rank-mu learning rates
            c1 = 2.0 / (((n_dim + 1.3) ** 2) + mu_eff_)
            alpha_mu = 2.0
            c_mu = min(
                1.0 - c1,
                alpha_mu
                * (mu_eff_ - 2.0 + 1.0 / mu_eff_)
                / (((n_dim + 2.0) ** 2) + alpha_mu * mu_eff_ / 2.0),
            )

            # commit back to cfg
            self.cfg.c_sigma = float(c_sigma)
            self.cfg.d_sigma = float(d_sigma)
            self.cfg.c_c = float(c_c)
            self.cfg.c1 = float(c1)
            self.cfg.c_mu = float(c_mu)
        return _State(
            X=X0,
            sigma=self.cfg.sigma0,
            B=B,
            D=D,
            p_sigma=p_sigma,
            p_c=p_c,
            weights=weights,
            mu_eff=mu_eff,
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
        codec = _CoordCodec(st.X)
        cands: list[Frame] = []
        self._last_steps = []
        for i in range(lam):
            U = codec.decode(y[:, i])
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
        cσ, dσ = cfg.c_sigma, cfg.d_sigma
        p_sigma_old = (1.0 - cσ) * st.p_sigma + np.sqrt(
            cσ * (2.0 - cσ) * st.mu_eff
        ) * y_white
        sigma_new = st.sigma * np.exp(
            (cσ / dσ) * (np.linalg.norm(p_sigma_old) / self._E_norm - 1.0)
        )

        # Covariance path (old chart)
        cc = cfg.c_c
        p_c_old = (1.0 - cc) * st.p_c + np.sqrt(cc * (2.0 - cc) * st.mu_eff) * y_bar

        # Covariance update (old chart)
        k = self.k
        C_old = st.B @ np.diag(st.D**2) @ st.B.T
        c1, cμ = cfg.c1, cfg.c_mu
        C_old *= 1.0 - c1 - cμ
        C_old += c1 * np.outer(p_c_old, p_c_old)

        # Positive weights
        for j in range(mu):
            yj = Y_sel[:, j]
            C_old += cμ * w_pos[j] * np.outer(yj, yj)

        # Active (negative) weights — PSD-safe scaling with warmup gating
        use_active_now = (
            cfg.use_active and (self._iter >= cfg.active_warmup) and (lam > mu)
        )
        if use_active_now:
            neg_idx = order[mu:]
            Y_neg = np.stack([self._last_steps[i] for i in neg_idx], axis=1)  # k×(λ-μ)
            w_neg = W[mu:]
            Yw = st.B.T @ Y_neg
            Yw = Yw / np.maximum(st.D[:, None], cfg.jitter)
            scales = 1.0 / np.maximum(np.sum(Yw**2, axis=0), cfg.jitter)
            for j in range(Y_neg.shape[1]):
                C_old += cμ * w_neg[j] * scales[j] * np.outer(Y_neg[:, j], Y_neg[:, j])

        # --- move the mean and build the new chart -------------------------
        codec_X = _CoordCodec(st.X)
        U_bar = codec_X.decode(y_bar)
        X_new = PRODUCT_CP.retract(st.X, st.sigma * U_bar)
        codec_Y = _CoordCodec(X_new)

        # --- build T (old -> new) via (d-1) parallel transports ------------
        n, d = st.X.shape
        m = n * (d - 1)
        k = 2 * m
        Qx = codec_X.Q_blocks
        Qy = codec_Y.Q_blocks

        V_cols: list[np.ndarray] = []
        for j in range(d - 1):
            Uj = np.empty((n, d), dtype=np.complex128)
            for i in range(n):
                Uj[i, :] = Qx[i][:, j]
            Vj = PRODUCT_CP.transport(st.X, X_new, Uj)  # (n, d)
            V_cols.append(Vj)

        blocks_real: list[np.ndarray] = []
        for i in range(n):
            Mi = np.stack([V_cols[j][i, :] for j in range(d - 1)], axis=1)  # (d, d-1)
            Ti_c = np.conj(Qy[i]).T @ Mi  # (d-1, d-1) complex
            Re, Im = Ti_c.real, Ti_c.imag
            Ti_real = np.block([[Re, -Im], [Im, Re]])  # 2(d-1) x 2(d-1)
            blocks_real.append(Ti_real)

        T = np.zeros((k, k), dtype=np.float64)
        off = 0
        r = 2 * (d - 1)
        for i in range(n):
            T[off : off + r, off : off + r] = blocks_real[i]
            off += r

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
        tol: float = 1e-20,
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
