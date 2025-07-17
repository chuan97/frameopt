import numpy as np

from evomof.core.energy import diff_coherence, grad_diff_coherence
from evomof.core.frame import Frame
from evomof.optim.local import polish_with_cg


def test_conjgrad_smoke() -> None:
    """One-step polish should not make things worse (basic sanity)."""
    f0 = Frame.random(8, 4, rng=np.random.default_rng(0))
    e0 = diff_coherence(f0, p=16)

    f1 = polish_with_cg(
        f0,
        energy_fn=lambda F: diff_coherence(F, p=16),
        grad_fn=lambda F: grad_diff_coherence(F, p=16),
        maxiter=5,
    )
    e1 = diff_coherence(f1, p=16)

    assert e1 <= e0
