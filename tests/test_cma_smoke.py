import numpy as np
import pytest

from evomof.core.frame import Frame
from evomof.optim.cma.projection import ProjectionCMA
from evomof.optim.cma.utils import frame_to_realvec, realvec_to_frame


def test_ask_returns_correct_population():
    """
    ProjectionCMA.ask() should return a list of Frame objects of length popsize,
    each with unit-norm vectors of shape (n, d).
    """
    n, d, popsize = 7, 4, 12
    algo = ProjectionCMA(
        n=n, d=d, sigma0=0.5, popsize=popsize, seed=123, energy_kwargs={"p": 4}
    )
    pop = algo.ask()
    # Check type and length
    assert isinstance(pop, list), "ask() should return a list"
    assert len(pop) == popsize, f"Expected population size {popsize}, got {len(pop)}"
    # Check each element is a Frame with correct shape and unit-norm columns
    for fr in pop:
        assert isinstance(fr, Frame), "Each element must be a Frame"
        vecs = fr.vectors
        # Shape check
        assert vecs.shape == (n, d)
        # Columns should be unit-norm
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-8), "Columns must be unit-norm"


def test_tell_allows_reinjection_and_continued_optimization():
    """
    After ask() and manual tell(), ProjectionCMA.step() should proceed without error.
    """
    n, d, popsize = 6, 3, 8
    # Use a simple energy proxy via energy_kwargs
    algo = ProjectionCMA(
        n=n, d=d, sigma0=0.3, popsize=popsize, seed=42, energy_kwargs={"p": 6}
    )
    # Sample a population
    population = algo.ask()
    # Provide dummy energies (e.g., all ones)
    energies = [1.0 for _ in population]
    # Reinjection should not raise
    algo.tell(population, energies)
    # After reinjection, step() should still work
    best_frame, best_energy = algo.step()
    assert isinstance(best_frame, Frame), "step() should return a Frame"
    assert isinstance(best_energy, float), "step() should return an energy float"
    # And best_energy should be finite
    assert np.isfinite(best_energy), "Energy must be finite"


def test_sigma_property_reflects_initial_and_updates():
    # Initialize with a known sigma0
    algo = ProjectionCMA(n=5, d=3, sigma0=0.123, popsize=4, seed=0)
    # Right after init, sigma should match sigma0
    assert algo.sigma == pytest.approx(0.123)
    # After one step, sigma should have been adapted (typically < sigma0)
    _ = algo.step()
    assert algo.sigma != pytest.approx(0.123)


def test_sigma_setter_updates_value():
    """
    Writing to ProjectionCMA.sigma should propagate to the underlying CMA-ES
    strategy and be readable back via the same property.
    """
    algo = ProjectionCMA(n=4, d=2, sigma0=0.1, popsize=6, seed=7)
    # Change sigma
    algo.sigma = 0.314
    # Read back and verify
    assert algo.sigma == pytest.approx(0.314), "Setter should update sigma"


def test_mean_property_roundtrip():
    """
    The mean property should allow round‑trip assignment of a custom vector.
    """
    n, d = 3, 2
    algo = ProjectionCMA(n=n, d=d, sigma0=0.2, popsize=5, seed=11)
    new_mean = np.full(2 * n * d, 0.123)  # simple deterministic vector
    algo.mean = new_mean
    assert np.allclose(algo.mean, new_mean), "Mean setter/getter round‑trip failed"


def test_mean_setter_shifts_offspring():
    """Changing ProjectionCMA.mean should steer the next offspring batch."""
    n, d, pop = 3, 2, 64
    algo = ProjectionCMA(n=n, d=d, sigma0=0.4, popsize=pop, seed=2025)

    m0 = algo.mean.copy()

    # shift mean far away
    new_mean = m0 + 5.0 * m0[::-1]
    algo.mean = new_mean

    # next generation
    xs_raw = algo._es.ask()
    m1 = np.mean(xs_raw, axis=0)

    assert np.linalg.norm(m1 - new_mean) < 0.1 * np.linalg.norm(m0 - new_mean)


def test_sigma_setter_scales_variance():
    """Boosting ProjectionCMA.sigma should widen the raw sampling cloud."""
    n, d, pop = 4, 3, 100
    algo = ProjectionCMA(n=n, d=d, sigma0=0.2, popsize=pop, seed=77)

    # --- baseline spread -----------------------------------------------
    xs0_raw = algo._es.ask()  # raw ambient vectors
    xs0 = np.vstack(xs0_raw)  # (pop, 2*n*d)
    std0 = xs0.std()

    # Tell CMA something so a second ask() is legal
    algo._es.tell(xs0_raw, [0.0] * pop)  # dummy fitness

    # --- boost sigma and resample --------------------------------------
    algo.sigma *= 3.0
    xs1_raw = algo._es.ask()
    xs1 = np.vstack(xs1_raw)
    std1 = xs1.std()

    # Expect clearly wider cloud (≥1.4×) but not insane
    assert std1 > 1.4 * std0, "Sigma boost did not increase spread enough"
    assert std1 < 3.8 * std0, "Spread exploded beyond expectation"


def test_tweak_persists_across_generation():
    """
    After mutating mean and sigma *post‑tell*, the very next raw sampling cloud
    should (a) be centred on the new mean and (b) exhibit a noticeably larger
    standard deviation.
    """
    algo = ProjectionCMA(n=3, d=2, sigma0=0.1, popsize=20, seed=999)

    # -- generation 0 ----------------------------------------------------
    pop0_raw = algo._es.ask()  # raw ambient
    frames0 = [realvec_to_frame(x, algo.n, algo.d) for x in pop0_raw]
    energies0 = [algo.energy_fn(fr) for fr in frames0]
    algo._es.tell(pop0_raw, energies0)

    baseline_std = np.vstack(pop0_raw).std()

    # Inject a large sigma and set a far‑away mean
    algo.sigma = 1.5
    algo.mean = np.full(2 * algo.n * algo.d, 2.0)

    # -- generation 1 ----------------------------------------------------
    pop1_raw = algo._es.ask()
    xs = np.vstack(pop1_raw)

    # Expect the sample mean ~ 2.0 (within 0.5) and std clearly larger
    assert np.allclose(xs.mean(), 2.0, atol=0.5), "Mean tweak did not propagate"
    assert xs.std() > 1.4 * baseline_std, "Sigma tweak did not widen spread"


def test_run_stops_on_tol():
    n, d = 4, 2
    algo = ProjectionCMA(n=n, d=d, sigma0=0.3, popsize=12, seed=123)

    # run with an intentionally loose tolerance
    best = algo.run(max_gen=200, tol=1e-2, log_every=0)

    # Assert we didn’t need all 200 generations
    assert algo._es.countiter < 200, "run() did not stop early on tol"
    assert isinstance(best, Frame)


def test_start_frame_is_used():
    n, d = 5, 3
    init = Frame.random(n, d, rng=np.random.default_rng(999))

    algo = ProjectionCMA(n=n, d=d, sigma0=0.4, popsize=10, seed=1, start_frame=init)
    # Mean right after init should equal flattened input
    assert np.allclose(algo.mean, frame_to_realvec(init))
