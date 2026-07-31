# Introduction

EnhancedBayesianNetworks.jl is a Julia package for building, reducing, and querying **enhanced
Bayesian networks** (eBNs) [straub_bayesian_2010](@cite): Bayesian networks extended with the
continuous and functional nodes of structural reliability analysis, and with **imprecision** —
interval probabilities and probability boxes — carried consistently from the inputs through to
the inference result.

## Why enhanced Bayesian networks

A classical Bayesian network [jensen2007bayesian](@cite) is a directed acyclic graph of
**discrete** random variables, each with a conditional probability table (CPT) given its
parents. That is expressive for categorical reasoning, but engineering models rarely stop there:
quantities are continuous, and the probabilities that matter — a component's failure probability,
say — are not tabulated in advance but *computed* from a physical model.

An enhanced Bayesian network closes that gap by admitting three kinds of node side by side:

- **discrete** nodes, whose CPT is known a priori;
- **continuous** nodes, holding a probability distribution;
- **functional** nodes, whose conditional table is *not* given but derived from the parents
  through one or more [UncertaintyQuantification.jl](https://github.com/JuliaUQ/UncertaintyQuantification.jl)
  models, evaluated as structural reliability problems
  [behrensdorf_uncertaintyquantificationjl_2023](@cite).

See the [Nodes](nodes.md) chapter for the full taxonomy.

## Precision and imprecision

Every quantity in the model may be **precise** or **imprecise**. A discrete CPT entry can be a
single probability or an `Interval`; a continuous node can hold an ordinary distribution or a
probability box [P_box_FAES](@cite). Imprecision expresses *epistemic* uncertainty — what is not
known well enough to pin down a single number [beer_imprecise_2013-1](@cite) — and it decides the
kind of network you end up with: any surviving imprecision turns a Bayesian network into a
**credal network** [cozman_credal_2000](@cite), a whole convex family of Bayesian networks rather
than a single one.

## From model to answer

Because a functional or continuous node cannot be queried directly, an eBN is first **reduced**
to a purely discrete network: its continuous nodes are discretized and its functional nodes are
evaluated as reliability problems, yielding a [`BayesianNetwork`](@ref) when everything stays
precise or a [`CredalNetwork`](@ref) when imprecision survives. This reduction is the core
operation of the library — see [Reduction & Reliability Analysis](reduction.md).

The reduced network is then ready for **inference**: exact posteriors by variable elimination for
a Bayesian network, or lower/upper posterior bounds for a credal one (see
[Inference](inference.md)). The package also supports learning CPTs from data
([Parameter Learning](parameterlearning.md)) and drawing the network structure
([Plotting](plotting.md)).

## How this manual is organized

- [Getting Started](gettingstarted.md) — install the package and run a first model.
- [Nodes](nodes.md) — the building blocks: discrete, continuous, and functional nodes.
- [Networks](networks.md) — Bayesian, credal, and enhanced Bayesian networks.
- [Reduction & Reliability Analysis](reduction.md) — reducing an eBN, and imprecise reliability.
- [Inference](inference.md) — variable elimination and credal inference.
- [Parameter Learning](parameterlearning.md) — learning CPTs from data.
- [Plotting](plotting.md) — visualizing a network.

The package builds directly on [UncertaintyQuantification.jl](https://github.com/JuliaUQ/UncertaintyQuantification.jl)
[behrensdorf_uncertaintyquantificationjl_2023](@cite), whose `Model`, `Parameter`,
`RandomVariable`, `Interval`, and `ProbabilityBox` types (and its simulation methods) are
re-exported and used throughout.
