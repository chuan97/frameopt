# evomof

![CI](https://github.com/chuan97/evomof/actions/workflows/ci.yaml/badge.svg)

A suite of optimization algorithms and tools for the numerical construction of maximally orthogonal frames (MOFs) in complex space, 
in particular Grassmannian frames/optimal packings.

Work in progress. 

As optimization algorithms, it currently features `ProjectionCMA`, a projection-based CMA-ES routine (based on [pycma](https://github.com/CMA-ES/pycma)), as a global optimizer, and `cg_minimize`, a conjugate gradient routine (based on [pymanop](https://github.com/pymanopt/pymanopt)), as a local optimizer. All optimizers are designed to work on the core object `Frame`. The frame coherence, potential and a differentiable surrogate for the coherence are implemented in `energy.py` and can be minimized with any of the optimizers.

The submodule `models` defines an API to build custom models to construct MOFs, a model takes a `Problem` (a pair of $(d, n)$ values indicating the dimension $d$ and the size $n$ of the desired MOF) as input and outputs a `Result`.  The `Model` protocol is designed to allow the construction and automated benchmarking of custom optimization pipelines with a common interface. For example, `ProjectionPRampModel` combines the `ProjectionCMA` optimizer with a ramp up protocol in the exponent $p$ of the energy function `diff_coherence`, with the goal of producing Grassmannian frames. A sequential combination of different optimizers, e.g. global then local, could also be implemented as a model.

```text
src
└── evomof
    ├── __init__.py
    ├── bounds.py
    ├── core
    │   ├── __init__.py
    │   ├── _types.py
    │   ├── energy.py
    │   ├── frame.py
    │   └── manifold.py
    ├── optim
    │   ├── cma
    │   │   ├── __init__.py
    │   │   ├── projection.py
    │   │   ├── riemannian.py
    │   │   └── utils.py
    │   ├── local
    │   │   ├── __init__.py
    │   │   └── cg.py
    │   └── utils
    │       └── p_scheduler.py
    └── py.typed
models
├── __init__.py
├── api.py
├── cg.py
├── cg_pramp.py
├── projection.py
├── projection_pramp.py
└── utils.py
```




