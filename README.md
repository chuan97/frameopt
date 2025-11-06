# frameopt

![CI](https://github.com/chuan97/frameopt/actions/workflows/ci.yaml/badge.svg)

A suite of optimization algorithms and tools for the numerical construction of maximally orthogonal frames (MOFs) in complex space [1]. 
In particular, it can be used to compute Grassmannian frames/optimal packings in complex projective space [2].

[The Game of Sloanes](https://github.com/gnikylime/GameofSloanes) hosts a leaderboard with the best known packings [2].

The core objects are:
* `frameopt.core.frame.Frame`: stores a finite unit-norm frame of $n$ elements in complex space $\mathbb C^d$. The global phase of each vector is removed, so `Frame` actually stores an element of $(\mathbb C P^{d-1})^n$.
* `frameopt.core.manifold.ProducCP`: implements operations on the product manifold of complex projective spaces, i.e. the manifold of frames as stored in `Frame`. Supported operations include: projection onto the tangent space at a point, retraction onto the manifold, and parallel transport of tangent vectors.
* `frameopt.core.manifold.chart`: implements a chart at a point, i.e. at a frame, in the product manifold of complex projective spaces. The chart provides an orthonormal basis in tangent space and can thus encode and decode tangent vectors to and from real coordinates in $\mathbb R^{2n(d-1)}$, and transport these coordinates across charts. It can also transport its basis to another point to form a new chart there. It is used by RiemannianCMA.

The available optimizers are:
* `frameopt.optim.cma.RiemannianCMA`: is an adaptation of CMA-ES as described in Ref. [3] to Riemannian manifolds. It interfaces classical CMA-ES with the manifold by sampling on the real space of chart coordinates for tangent vectors at the mean frame, generating candidates and a new mean by retraction, and transporting evolution paths and the covariance matrix by parallel transport.
* `frameopt.optim.cma.ProjectionCMA`: is a simpler implementation of CMA-ES on the product manifold of complex projective spaces by sampling in real ambient space and projecting candidates to a frame before evaluating the energy. It uses [pycma](https://github.com/CMA-ES/pycma) as backend.
* `frameopt.optim.local.cg_minimize`: implements the conjugate gradient method on the product manifold of complex projective spaces. It uses [pymanopt](https://github.com/pymanopt/pymanopt) as backend.
* `frameopt.optim.local.tr_minimize`: implements the trust region method on the product manifold of complex projective spaces. It uses [pymanopt](https://github.com/pymanopt/pymanopt) as backend.

Additionally:
* `frameopt.core.energy`: implements functions to compute the frame coherence, the $p$-frame potential and two [smooth maximum](https://en.wikipedia.org/wiki/Smooth_maximum) surrogates for the coherence, the $p$-norm wich is just the $p$-th root of the $p$-frame potential and mellowmax, as well as their gradients. These energy functions can be minimized with any of the optimizers.
* `frameopt.bounds`: implements the Buhk-Cox, Welch, orthoplex, and Levenstein bounds for the coherence of a frame [2], as well as a master function `max_lower_bound` that returns the maximum applicable lower bound for each $(d, n)$ pair.
* `frameopt.model`: defines an API to build custom models to construct MOFs. A model takes a `frameopt.model.api.Problem` (a pair of $(d, n)$ values indicating the dimension $d$ and the size $n$ of the desired frame) as input and outputs a `frameopt.model.api.Result`.  The `frameopt.model.api.Model` protocol is designed to facilitate the construction and automated benchmarking of custom optimization pipelines with a common interface.
* `models/`: contains some models. For example, `models/projection_pramp.py` implements `ProjectionPRampModel`, which combines the `ProjectionCMA` optimizer with a ramp up protocol in the exponent $p$ of the differentiable surrogate for the coherence, with the goal of producing Grassmannian frames. The ramp up protocol is provided by `frameopt.model.p_scheduler.AdaptivePScheduler`.
* `scripts/`: contains some examples and some useful scripts. In particular `scripts/bench/run_model.py` can read a set of input problems, and a model and its parameters from config files, parallel run the model on the problems, and store the results.

## Project structure
```text
src
└── frameopt
    ├── __init__.py
    ├── bounds.py
    ├── core
    │   ├── __init__.py
    │   ├── _types.py
    │   ├── energy.py
    │   ├── frame.py
    │   └── manifold.py
    ├── model
    │   ├── __init__.py
    │   ├── api.py
    │   ├── p_scheduler.py
    │   └── utils.py
    ├── optim
    │   ├── __init__.py
    │   ├── cma
    │   │   ├── __init__.py
    │   │   ├── projection.py
    │   │   ├── riemannian.py
    │   │   └── utils.py
    │   └── local
    │       ├── __init__.py
    │       ├── _pymanopt_adapters.py
    │       ├── cg.py
    │       └── tr.py
    └── py.typed
models
├── cg.py
├── cg_pramp.py
├── projection.py
├── projection_cg.py
├── projection_pramp.py
├── riemannian.py
├── riemannian_pramp.py
├── tr.py
└── tr_pramp.py
scripts
├── _utils.py
├── bench
│   ├── quick_energy_benchmark.py
│   └── run_model.py
├── cli
│   ├── coherence_from_txt.py
│   └── verify_frame.py
└── examples
    ├── cma_projection.py
    └── cma_riemannian.py
```

## References

[1] S. Roca-Jerat, J. Román-Roche, Mach. Learn.: Sci. Technol. 6 035022 (2025)

[2] J. Jasper, E. J. King, D. G. Mixon, arXiv:1907.07848 (2019)

[3] N. Hansen, arXiv:1604.00772 (2023)




