import numpy as np

from frameopt.core.energy import grad_pnormmax_coherence, pnormmax_coherence
from frameopt.core.frame import Frame
from frameopt.optim.local.tr import minimize as tr_minimize


def test_tr_smoke() -> None:
    """One-step polish should not make things worse (basic sanity)."""
    rng = np.random.default_rng(0)
    f0 = Frame.random(8, 4, rng=rng)
    e0 = pnormmax_coherence(f0, p=16)

    f1 = tr_minimize(
        f0,
        energy_fn=lambda F: pnormmax_coherence(F, p=16),
        grad_fn=lambda F: grad_pnormmax_coherence(F, p=16),
        maxiter=5,
        verbosity=0,
    )
    e1 = pnormmax_coherence(f1, p=16)

    assert e1 <= e0 + 1e-12
