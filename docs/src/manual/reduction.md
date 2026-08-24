# Reduction & Structural Reliability Problem

An [Enhanced Bayesian Network](@ref) (eBN) mixes discrete, continuous, and functional nodes, so it cannot be queried directly. 
[`reduce`](@ref) turns it into an inference-ready **discrete** network, a [Bayesian Network](@ref) (BN) when everything stays precise, or a [Credal Network](@ref) (CN) when any imprecision is present. 
This is the central operation of the library: it is the route by the **Structural Reliability Methods** selected to solve the **Structural Reliability Problems** (SRPs) that defines the *Conditional Probability Tables* (CPTs) of each functional node of the network.

## The reduction pipeline

[`reduce`](@ref) follows the enhanced Bayesian Network (eBN) procedure of [straub_bayesian_2010](@cite), and rests on the node-removal operations for evaluating influence diagrams [Shachter86a](@cite):

1. **Build** and **Order** the network ([`order!`](@ref), see [Building and validating a network](@ref)).
2. **Discretize** every continuous node that carries a discretization structure it is replaced by a discrete node (the per-interval probability masses) plus a residual continuous node, with parents rewired to the discrete part and children to the continuous part (`discretize!`, see [Discretization](@ref)).
3. **Transfer** the model of each continuous functional node without discretization, into its children (`_transfer_continuous_functional_node!`). 
This is a computational optimization of the evaluation stage. 
A continuous-functional node that merely feeds another functional node would otherwise be evaluated on its own, sampling its models and fitting an [`EmpiricalDistribution`](@extref `UncertaintyQuantification.EmpiricalDistribution`) per scenario, only for that distribution to be re-sampled by the child and then thrown away when the node is eliminated. 
Instead, its models are *prepended* to the child's model chain, so the samples already drawn are propagated straight through the child during the child's single Structural Reliability Problem evaluation. 
The intermediate empirical distribution is never built for a node that reduction removes anyway.
4. **Evaluate** functional nodes in dependency order. A node whose parents are all non-functional has ready inputs: its conditional table is filled by solving a Structural Reliability Problem over the scenario grid of its [`discrete_ancestors`](@ref). The functional node is then replaced by a node with defined CPT, whose kind and precision mirror what was evaluated (see [Precise and imprecise outcomes](@ref) below).
5. **Eliminate** continuous parents. A continuous node that fed only the just-evaluated node is removed, and its parents reconnected to its children (`_eliminate_node!`: node removal [Shachter86a](@cite)); one that still feeds other functional nodes keeps its remaining edges, and only the spent edge is cut.
6. **Dispatch** to the concrete type: once no node is continuous, an all-precise network becomes a BN and a network with any surviving imprecision a CN.

!!! tip "Progress bar"
    Step 4 is the expensive part — one Structural Reliability Problem per scenario. [`reduce`](@ref) takes a `progress` keyword that shows a progress bar over each functional node's scenario grid as it is evaluated. It defaults to `isinteractive()` (shown in the REPL, silent in scripts, tests, and this documentation); pass `progress = true` or `progress = false` to force it.

## Structural Reliability Problems

The heart of step 4 is the **structural reliability problem** (SRP). Each functional node's parents' uncertainty or imprecision are propagated through its
models. For a discrete-functional node, the node's `performance` function splits the outcome into a *failed* region (`performance < 0`) and a *safe* one, whose probability estimate becomes the node's conditional table. 
The`simulation` attached to the node decides *how* that probability is estimated.

### Precise inputs

When every input is precise, the SRP is solved with a **single-loop** Structural Reliability Method, such as `MonteCarlo`, [`FORM`](@extref `UncertaintyQuantification.FORM`), or advanced Monte Carlo methods ([`SubSetSimulation`](@extref `UncertaintyQuantification.SubSetSimulation`), `LineSampling`, and the like). 
Each scenario yields a single failure probability, so the reduced node is precise and a BN is returned.

```@example reduction
using EnhancedBayesianNetworks

Load = DiscreteNode(:Load, [:low => [Parameter(1.0, :Load)], :high => [Parameter(3.0, :Load)]])
Load[:Load => :low] = 0.7; Load[:Load => :high] = 0.3
R = ContinuousNode(:R, Normal(3.0, 0.5))                 # precise resistance
model = Model(df -> df.R .- df.Load, :g)                 # limit state g = R - Load
F = DiscreteFunctionalNode(:F, [model], df -> df.g, MonteCarlo(2000))

ebn = EnhancedBayesianNetwork([Load, R, F])
add_child!(ebn, :Load, :F)
add_child!(ebn, :R, :F)
order!(ebn)
ebn
```
```@example reduction
reduced = reduce(ebn)                                    # -> BayesianNetwork
```

### Imprecise inputs

When an input is imprecise: a continuous parent given as an [`Interval`](@extref `UncertaintyQuantification.Interval`) or [`ProbabilityBox`](@extref `UncertaintyQuantification.ProbabilityBox`). Then the failure probability is no longer a single number but a probability **interval** [beer_imprecise_2013-1](@cite). 
It is estimated by a **double-loop** structural reliability method — [`DoubleLoop`](@extref `UncertaintyQuantification.DoubleLoop`), or [`RandomSlicing`](@extref `UncertaintyQuantification.RandomSlicing`). 
Each scenario then produces a lower and an upper failure probability, the reduced node carries interval CPT entries, and [`reduce`](@ref) returns a [`CredalNetwork`](@ref).

Only the *type of the input* and the *simulation* change — the network is built exactly as before:

```@example reduction
Load = DiscreteNode(:Load, [:low => [Parameter(1.0, :Load)], :high => [Parameter(3.0, :Load)]])
Load[:Load => :low] = 0.7 
Load[:Load => :high] = 0.3
R = ContinuousNode(:R, Interval(2.0, 4.0))               # imprecise resistance
model = Model(df -> df.R .- df.Load, :g)
F = DiscreteFunctionalNode(:F, [model], df -> df.g, DoubleLoop(MonteCarlo(1000)))

ebn = EnhancedBayesianNetwork([Load, R, F])
add_child!(ebn, :Load, :F)
add_child!(ebn, :R, :F)
order!(ebn)
ebn
```

```@example reduction
reduced = reduce(ebn)                                    # -> CredalNetwork
```

The imprecision then flows straight into inference: querying the reduced credal network returns a CN with lower and upper bounds (see the [Inference](inference.md) chapter).

## Precise and imprecise outcomes

The kind and precision of the node produced in step 4 mirrors the functional node it replaces:

- a [`DiscreteFunctionalNode`](@ref) becomes a discrete node. Each scenario contributes a failure probability and its complement. In the **precise** case these are real numbers, so the reduced node has a real-valued CPT, and it is precise; in the **imprecise** case the failure probability comes out as a probability interval, so the CPT carries at least one interval entry and the node is imprecise.

- a [`ContinuousFunctionalNode`](@ref) becomes a continuous node whose distribution is refit from the drawn samples. 
In the **precise** case each scenario yields a single empirical distribution; in the **imprecise** case each scenario yields *two*, a lower-bound and an upper-bound empirical distribution that bracket the family of admissible distributions. Consolidating this lower/upper pair into a single p-box per scenario is ongoing work.