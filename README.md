# evomof

![CI](https://github.com/chuan97/evomof/actions/workflows/ci.yml/badge.svg)

Optimization tools for the numerical construction of maximally orthogonal frames (MOFs), 
in particular Grassmannian frames/optimal packings in complex space.

Work in progress. 

It currently features a projection-based CMA-ES routine (using [pycma](https://github.com/CMA-ES/pycma)) on the Grassmannian manifold as a global optimizer
and a conjugate gradient routine (based on [pymanops](https://github.com/pymanopt/pymanopt)) on the Grassmannian manifold as a local optimizer.


