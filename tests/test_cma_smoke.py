import numpy as np

from evomof.core.frame import Frame
from evomof.optim.cma.projection import ProjectionCMA


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
