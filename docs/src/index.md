# EnhancedBayesianNetworks.jl

```@raw html
<p align="center">
  <img src="./assets/logo.png" alt="EnhancedBayesianNetworks.jl logo" width="220"/>
</p>
```

A Julia package for building, reducing, and querying **enhanced Bayesian networks** — Bayesian
networks extended with the continuous and functional nodes of structural reliability analysis, and
with imprecision carried consistently from the inputs through to the inference result.

## Features

Current functionality includes:

* Node types
  * Discrete and continuous nodes, with conditional probability tables or distributions known a priori
  * Discrete and continuous *functional* nodes, whose tables are derived from the parents through [UncertaintyQuantification.jl](https://github.com/JuliaUQ/UncertaintyQuantification.jl) models
  * Imprecision at every level — interval probabilities and probability boxes
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

---

## Installation

EnhancedBayesianNetworks.jl is not yet registered. Install the latest version directly from GitHub through the Julia package manager:

```julia
julia> ]add https://github.com/JuliaUQ/EnhancedBayesianNetworks.jl
julia> using EnhancedBayesianNetworks
```

New here? Start with the [Introduction](manual/introduction.md) for the concepts, or jump to [Getting Started](manual/gettingstarted.md) to run your first model.

---

## Related packages

* [UncertaintyQuantification.jl](https://github.com/JuliaUQ/UncertaintyQuantification.jl): the structural-reliability and uncertainty-propagation backbone that EnhancedBayesianNetworks.jl builds on — its models, inputs, and simulation methods evaluate every functional node.
