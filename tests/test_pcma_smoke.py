import numpy as np
import pytest

from frameopt.core.frame import Frame
from frameopt.optim.cma import ProjectionCMA
from frameopt.optim.cma.utils import frame_to_realvec, realvec_to_frame


def test_ask_returns_correct_population():
    """
    ProjectionCMA.ask() should return a list of Frame objects of length popsize,
    each with unit-norm vectors of shape (n, d).
    """
    n, d, popsize = 7, 4, 12
    algo = ProjectionCMA(n=n, d=d, sigma0=0.5, popsize=popsize, seed=123)
    raws = algo.ask()
    pop = [realvec_to_frame(x, n, d) for x in raws]
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
    algo = ProjectionCMA(n=n, d=d, sigma0=0.3, popsize=popsize, seed=42)
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


def test_frame_to_realvec_and_back():
    n, d = 6, 4
    frame = Frame.random(n, d, rng=np.random.default_rng(2024))
    vec = frame_to_realvec(frame)
    reconstructed = realvec_to_frame(vec, n, d)
    assert np.allclose(frame.vectors, reconstructed.vectors)
