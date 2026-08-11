# Reduction & Structural Reliability Problem

An [`EnhancedBayesianNetwork`](@ref) mixes discrete, continuous, and functional nodes, so it
cannot be queried directly. [`reduce`](@ref) turns it into an inference-ready **discrete**
network — a [`BayesianNetwork`](@ref) when everything stays precise, or a [`CredalNetwork`](@ref)
when any imprecision survives. This is the central operation of the library: it is the route by
which structural reliability analysis enters the Bayesian-network formalism
[straub_bayesian_2010](@cite), and the mechanism that carries imprecision from the physical
inputs through to the inference result.

## The reduction pipeline

[`reduce`](@ref) follows the enhanced-Bayesian-network procedure of
[straub_bayesian_2010](@cite), and rests on the node-removal operations for evaluating influence
diagrams [Shachter86a](@cite):

1. **Order** the network ([`order!`](@ref)).
2. **Discretize** every continuous node that carries a [discretization strategy](nodes.md) — it is
   replaced by a discrete surrogate (the per-interval probability masses) plus a residual
   continuous node, with parents rewired to the discrete part and children to the continuous
   part (`discretize!`).
3. **Transfer** each continuous-functional node's models into its children
   (`_transfer_continuous_functional_node!`). This is a computational optimization of the
   evaluation stage. A continuous-functional node that merely feeds another functional node
   would otherwise be evaluated on its own — sampling its models and fitting an
   `EmpiricalDistribution` from [UncertaintyQuantification.jl](https://github.com/JuliaUQ/UncertaintyQuantification.jl)
   per scenario — only for that distribution to be re-sampled by the child and then thrown away
   when the node is eliminated. Instead, its models are *prepended* to the child's model chain,
   so the samples already drawn for the child are propagated straight through them during the
   child's single structural reliability problem. The intermediate empirical distribution is
   never built for a node that reduction removes anyway.
4. **Evaluate** functional nodes in dependency order. A node whose parents are all
   non-functional has ready inputs: its conditional table is filled by solving a structural
   reliability problem over the scenario grid of its [`discrete_ancestors`](@ref), using the
   node's simulation [behrensdorf_uncertaintyquantificationjl_2023](@cite). The functional node
   is then replaced by a plain node whose kind and precision mirror what was evaluated (see
   [Precise and imprecise outcomes](@ref) below).
5. **Eliminate** continuous parents. A continuous node that fed only the just-evaluated node
   is removed and its parents reconnected to its children (`_eliminate_node!` — node removal
   [Shachter86a](@cite)); one that still feeds other functional nodes keeps its remaining
   edges, and only the spent edge is cut.
6. **Dispatch** to the concrete type: once no node is continuous, an all-precise network
   becomes a [`BayesianNetwork`](@ref) and a network with any surviving imprecision a
   [`CredalNetwork`](@ref).

## Structural reliability problems

The heart of step 4 is the **structural reliability problem** (SRP). Each functional node's
parents' uncertainty is propagated through its
[UncertaintyQuantification.jl](https://github.com/JuliaUQ/UncertaintyQuantification.jl) models,
and — for a [`DiscreteFunctionalNode`](@ref) — the node's `performance` function splits the
outcome into a *failed* region (`performance < 0`) and a *safe* one, whose probability estimate
becomes the node's conditional table [behrensdorf_uncertaintyquantificationjl_2023](@cite). The
`simulation` attached to the node decides *how* that probability is estimated, and whether the
result is precise or imprecise.

### Precise inputs

When every input is precise, the SRP is solved with a standard reliability simulation —
`MonteCarlo`, [`SubSetSimulation`](@extref `UncertaintyQuantification.SubSetSimulation`),
`LineSampling`, and the like. Each scenario yields a single
failure probability, so the reduced node is precise and [`reduce`](@ref) returns a
[`BayesianNetwork`](@ref).

```julia
using EnhancedBayesianNetworks

Load = DiscreteNode(:Load, [:low => [Parameter(1.0, :Load)], :high => [Parameter(3.0, :Load)]])
Load[:Load => :low] = 0.7; Load[:Load => :high] = 0.3
R = ContinuousNode(:R, Normal(3.0, 0.5))                 # precise resistance
model = Model(df -> df.R .- df.Load, :g)                 # limit state g = R - Load
F = DiscreteFunctionalNode(:F, [model], df -> df.g, MonteCarlo(2000))

ebn = EnhancedBayesianNetwork([Load, R, F])
add_child!(ebn, :Load, :F); add_child!(ebn, :R, :F); order!(ebn)

reduced = reduce(ebn)                                    # -> BayesianNetwork
```

### Imprecise inputs

When any input is imprecise — a continuous parent given as an `Interval` or `ProbabilityBox`
[P_box_FAES](@cite), or an interval-valued discrete parent — the failure probability is no
longer a single number but an **interval** [beer_imprecise_2013-1](@cite). It is estimated by an
*outer* search over the imprecise inputs wrapping an *inner* reliability simulation: a
[`DoubleLoop`](@extref `UncertaintyQuantification.DoubleLoop`), or the more efficient
[`RandomSlicing`](@extref `UncertaintyQuantification.RandomSlicing`)
[behrensdorf_uncertaintyquantificationjl_2023](@cite). Each scenario then produces a lower and an
upper failure probability, the reduced node carries interval CPT entries, and [`reduce`](@ref)
returns a [`CredalNetwork`](@ref).

Only the *type of the input* and the *simulation* change — the network is built exactly as
before:

```julia
using EnhancedBayesianNetworks

Load = DiscreteNode(:Load, [:low => [Parameter(1.0, :Load)], :high => [Parameter(3.0, :Load)]])
Load[:Load => :low] = 0.7; Load[:Load => :high] = 0.3
R = ContinuousNode(:R, Interval(2.0, 4.0))               # imprecise resistance
model = Model(df -> df.R .- df.Load, :g)
F = DiscreteFunctionalNode(:F, [model], df -> df.g, DoubleLoop(MonteCarlo(1000)))

ebn = EnhancedBayesianNetwork([Load, R, F])
add_child!(ebn, :Load, :F); add_child!(ebn, :R, :F); order!(ebn)

reduced = reduce(ebn)                                    # -> CredalNetwork
```

The imprecision then flows straight into inference: querying the reduced credal network returns a
[`CredalPosterior`](@ref) with lower and upper bounds (see the [Inference](inference.md)
chapter).

```julia
infer(reduced, :F, Evidence(:Load => :high))            # CredalPosterior: [lower, upper]
```

## Precise and imprecise outcomes

The kind and precision of the node produced in step 4 mirror the functional node it replaces:

- a [`DiscreteFunctionalNode`](@ref) becomes a [`DiscreteNode`](@ref). Each scenario contributes
  a failure probability and its complement. In the **precise** case these are real numbers, so
  the reduced node has a real-valued CPT and stays precise; in the **imprecise** case the failure
  probability comes out as an `Interval`, so the CPT carries at least one interval entry and the
  node stays imprecise.
- a [`ContinuousFunctionalNode`](@ref) becomes a [`ContinuousNode`](@ref) whose distribution is
  refit from the drawn samples. In the **precise** case each scenario yields a single
  `EmpiricalDistribution`; in the **imprecise** case each scenario yields *two* — a lower-bound
  and an upper-bound `EmpiricalDistribution` that bracket the family of admissible distributions.
  (Consolidating this lower/upper pair into a single `ProbabilityBox` [P_box_FAES](@cite) per
  scenario is ongoing work.)

Whenever any evaluated node comes out imprecise, the final [`reduce`](@ref) dispatch yields a
[`CredalNetwork`](@ref) rather than a [`BayesianNetwork`](@ref) — imprecision at any input
propagates all the way to the reduced network.
