from evomof.core.energy import coherence
from evomof.optim.cma.projection import ProjectionCMA

n, d = 9, 3  # classic Grassmannian frame size
cma_proj = ProjectionCMA(n, d, sigma0=0.3, popsize=40, p=16)
best = cma_proj.run(max_gen=100)
print("Final coherence:", coherence(best), "Expected coherence:", 0.5)
