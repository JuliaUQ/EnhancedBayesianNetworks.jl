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

## The system probability distribution

!!! note "Notation"
    Throughout the manual, ``Y`` denotes a **discrete** node, ``X`` a **continuous** node, and
    ``Z`` a node of **either** kind. An uppercase symbol (``Y_i``, ``X_i``, ``Z_i``) is the node —
    a random variable — while the matching lowercase symbol (``y_i``, ``x_i``, ``z_i``) is one of
    its realizations. **Bold** marks a collection: a *set* of nodes (``\mathbf{Y}``, ``\mathbf{X}``)
    or a full joint assignment (``\mathbf{y}``, ``\mathbf{x}``). A node's parents, children, and
    spouses (co-parents) are written ``\mathrm{Pa}(\cdot)``, ``\mathrm{Ch}(\cdot)``, and
    ``\mathrm{Sp}(\cdot)``.

A Bayesian network encodes a joint distribution that *factorizes* over the graph. By the local
Markov property each variable is conditionally independent of its non-descendants given its
parents, so the joint distribution of ``n`` discrete variables ``Y_1, \dots, Y_n`` is the product
of the per-node conditional tables [jensen2007bayesian](@cite):

```math
p(y_1, \dots, y_n) = \prod_{i=1}^{n} p\!\left(y_i \mid \mathrm{Pa}(Y_i)\right),
```

where ``\mathrm{Pa}(Y_i)`` denotes the parents of ``Y_i``.

An enhanced Bayesian network keeps this factorized structure but admits both a set of
**discrete** nodes ``\mathbf{Y} = \{Y_1, \dots, Y_{n_Y}\}`` and a set of **continuous** nodes
``\mathbf{X} = \{X_1, \dots, X_{n_X}\}``. Its system joint distribution is then
the combined measure of the discrete probability mass functions and the continuous probability
density functions [straub_bayesian_2010](@cite):

```math
p(\mathbf{y} \mid \mathbf{x})\, f(\mathbf{x}) =
\prod_{Y_i \in \mathbf{Y}} p\!\left(y_i \mid \mathrm{Pa}(Y_i)\right)
\prod_{X_i \in \mathbf{X}} f\!\left(x_i \mid \mathrm{Pa}(X_i)\right),
```

where the ``p(\cdot)`` are the conditional PMFs of the discrete nodes and the ``f(\cdot)`` the
conditional PDFs of the continuous ones. Functional nodes are exactly the factors of this product
whose conditional table is *not* given a priori: it is obtained, during reduction, by solving a
structural reliability problem (see [Reduction & Reliability Analysis](reduction.md)).

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
[behrensdorf_uncertaintyquantificationjl_2023](@cite), whose [`Model`](@extref `UncertaintyQuantification.Model`),
[`Parameter`](@extref `UncertaintyQuantification.Parameter`),
[`RandomVariable`](@extref `UncertaintyQuantification.RandomVariable`),
[`Interval`](@extref `UncertaintyQuantification.Interval`), and
[`ProbabilityBox`](@extref `UncertaintyQuantification.ProbabilityBox`) types (and its simulation
methods) are re-exported and used throughout.
