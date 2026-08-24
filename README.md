# EnhancedBayesianNetworks.jl

[![CI](https://github.com/JuliaUQ/EnhancedBayesianNetworks.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaUQ/EnhancedBayesianNetworks.jl/actions/workflows/ci.yml)
[![Coverage Status](https://codecov.io/github/JuliaUQ/EnhancedBayesianNetworks.jl/graph/badge.svg?token=10uJhq58fJ)](https://codecov.io/github/JuliaUQ/EnhancedBayesianNetworks.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14054153.svg)](https://doi.org/10.5281/zenodo.14054153)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliauq.github.io/EnhancedBayesianNetworks.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliauq.github.io/EnhancedBayesianNetworks.jl/dev)

A Julia package for imprecise enhanced Bayesian Networks. Current functionality includes:


* Node types
  * Discrete and continuous nodes, with conditional probability tables or distributions known a priori
  * Discrete and continuous *functional* nodes, whose tables are derived from the parents through [UncertaintyQuantification.jl](https://github.com/JuliaUQ/UncertaintyQuantification.jl) models
  * Imprecision at *credal* and *simulation* level — interval probabilities and probability boxes
* Network types
  * Bayesian networks
  * Credal networks (imprecise)
  * Enhanced Bayesian networks — discrete, continuous, and functional nodes side by side
* Reduction & reliability analysis
  * Discretization of continuous nodes
  * Evaluation of functional nodes as structural reliability problems (Monte Carlo, Subset Simulation, Line Sampling, …)
  * Imprecise reliability by Double Loop and Random Slicing
  * Precise inputs reduce to a Bayesian network, imprecise inputs to a credal network
* Inference
  * Exact inference by variable elimination
  * Credal inference with lower/upper posterior bounds
* Parameter learning
  * Maximum likelihood estimation from complete data
  * Expectation–Maximization for data with missing entries
* Visualization
  * Layered, top-down network plots that encode each node's kind, precision, and discretization
