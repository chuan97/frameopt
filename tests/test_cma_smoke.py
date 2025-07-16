from evomof.optim.cma.projection import ProjectionCMA


def test_cma_smoke():
    algo = ProjectionCMA(6, 3, sigma0=0.3, popsize=10, p=8, rng=0)
    best, energy = algo.step()  # one generation
    assert energy < 2.0  # arbitrary finite value
