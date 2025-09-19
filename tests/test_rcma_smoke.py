import numpy as np

from evomof.core.energy import diff_coherence
from evomof.core.frame import Frame
from evomof.core.manifold import PRODUCT_CP

# Import the user-facing RCMA class (wrapper around the engine)
from evomof.optim.cma.riemannian import RiemannianCMA


def test_ask_returns_correct_population():
    """
    RiemannianCMA.ask() should return a list of Frame objects of length popsize,
    each with unit-norm rows of shape (n, d).
    """
    n, d, popsize = 7, 4, 12
    algo = RiemannianCMA(n=n, d=d, sigma0=0.25, popsize=popsize, seed=123)

    pop = algo.ask()

    # Check type and length
    assert isinstance(pop, list), "ask() should return a list"
    assert len(pop) == popsize, f"Expected population size {popsize}, got {len(pop)}"

    # Check each element is a Frame with correct shape and unit-norm rows
    for fr in pop:
        assert isinstance(fr, Frame), "Each element must be a Frame"
        vecs = fr.vectors
        assert vecs.shape == (n, d)
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-8), "Rows must be unit-norm"


def test_tell_runs_without_error():
    """
    After ask() and manual tell(), step() should proceed without error.
    """
    n, d, popsize = 6, 3, 10
    algo = RiemannianCMA(
        n=n,
        d=d,
        sigma0=0.2,
        popsize=popsize,
        seed=42,
        energy_fn=diff_coherence,
        energy_kwargs={"p": 6},
    )

    population = algo.ask()
    # Provide dummy energies (e.g., all ones) — we only test plumbing
    energies = [1.0 for _ in population]

    # Reinjection should not raise
    algo.tell(population, energies)

    # After reinjection, step() should still work
    best_frame, best_energy = algo.step()
    assert isinstance(best_frame, Frame), "step() should return a Frame"
    assert isinstance(best_energy, float), "step() should return an energy float"
    assert np.isfinite(best_energy), "Energy must be finite"


def test_run_returns_finite():
    """A short run should return a finite best value and a Frame."""
    n, d = 4, 3
    algo = RiemannianCMA(
        n=n,
        d=d,
        sigma0=0.3,
        popsize=12,
        seed=7,
        energy_fn=diff_coherence,
        energy_kwargs={"p": 4},
    )

    # run a few generations with a loose tolerance
    best_frame = algo.run(max_gen=25, tol=1e-3, log_every=0)

    assert isinstance(best_frame, Frame)
    assert np.isfinite(diff_coherence(best_frame))


def test_start_frame_is_used_with_small_sigma():
    """
    If we supply a start_frame and very small sigma0, the first ask() population
    should stay very close to that start frame (intrinsic sampling).
    """
    n, d = 5, 3
    rng = np.random.default_rng(999)
    init = Frame.random(n, d, rng=rng)

    # Tiny sigma → samples should be near init on the manifold
    algo = RiemannianCMA(
        n=n,
        d=d,
        sigma0=1e-4,
        popsize=16,
        seed=1234,
        start_frame=init,
        energy_fn=diff_coherence,
    )

    pop = algo.ask()

    # Measure average geodesic distance (via log map norm)
    dists = []
    for fr in pop:
        xi = PRODUCT_CP.log_map(init, fr)
        dists.append(np.linalg.norm(xi))
    mean_dist = float(np.mean(dists))

    assert mean_dist < 1e-2, "Initial population not concentrated around start_frame"


def test_energy_decreases_on_run():
    """
    After a few generations, the best energy should improve vs. the first
    generation's best (stochastic, so use a modest relative threshold).
    """
    n, d, pop = 6, 3, 20
    algo = RiemannianCMA(
        n=n,
        d=d,
        sigma0=0.3,
        popsize=pop,
        seed=321,
        energy_fn=diff_coherence,
        energy_kwargs={"p": 4},
    )

    # Generation 0: sample and evaluate once to get a baseline best
    pop0 = algo.ask()
    E0 = [diff_coherence(fr, p=4) for fr in pop0]
    best0 = float(np.min(E0))
    algo.tell(pop0, np.array(E0))

    # Run a few generations
    best_frame = algo.run(max_gen=30, tol=0.0, log_every=0)
    best_after = float(diff_coherence(best_frame, p=4))

    # Expect at least a small improvement (5%)
    assert best_after <= 0.95 * best0, (
        f"Energy did not improve enough: initial best {best0:.3e}, "
        f"after run {best_after:.3e}"
    )
