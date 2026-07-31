# Nodes

Nodes are the fundamental building blocks of a network: together with the edges, they
graphically represent the random variables of the network. Every node carries a name (a
`Symbol`) that uniquely identifies it, a conditional probability table (CPT), which 
might by a priori **known** or **unknown**, and **continuous** or **discrete**. 
A node is classified along two independent lines.

**By position in the graph** a node is either a **root** — no parents, its state depending
on nothing else — or a **child**, with one or more parents that influence it. This is a
property of the assembled network, read back with [`isroot`](@ref), not a separate type.

**By the nature of its variable** a node is either **discrete** — a finite set of mutually
exclusive states (discrete CPT) — or **continuous** — a real-valued quantity described by a probability
distribution (continuous CPT).

Cutting across both is the distinction that decides *which type you actually construct*:

> **Is the node's conditional probability table (CPT) known a priori?**
>
> - **Yes** — you supply it explicitly, as a [`DiscreteNode`](@ref) or a
>   [`ContinuousNode`](@ref).
> - **No** — the CPT is instead defined by a *functional relationship* with the parents,
>   so the node **must be functional** ([`DiscreteFunctionalNode`](@ref),
>   [`ContinuousFunctionalNode`](@ref)). Its table does not exist up front: it is
>   materialized only when the network is reduced, by propagating the parents' uncertainty
>   through the node's models (structural reliability problems).

| Type | Variable | CPT | Role |
|:--|:--|:--|:--|
| [`DiscreteNode`](@ref) | discrete | known a priori | root or child |
| [`ContinuousNode`](@ref) | continuous | known a priori | root or child |
| [`DiscreteFunctionalNode`](@ref) | discrete | **not** known a priori | child only |
| [`ContinuousFunctionalNode`](@ref) | continuous | **not** known a priori | child only |

The distributions and uncertainty models a node is built from — `Parameter`,
`RandomVariable`, `Interval`, `ProbabilityBox`, `Model`, and the simulation types such as
`MonteCarlo` — come from
[UncertaintyQuantification.jl](https://github.com/JuliaUQ/UncertaintyQuantification.jl)
[behrensdorf_uncertaintyquantificationjl_2023](@cite) and are re-exported by
EnhancedBayesianNetworks, so they are available without a separate `using`.

Entries may also be **precise** or **imprecise**: a discrete entry is a `Real`
probability for a **precise** discrete node or an `Interval` for an **imprecise** 
(or **credal**) discrete node [weichselberger_theory_2000, Levi1980-LEVTEO-7](@cite); a continuous entry is a `UnivariateDistribution` for 
a **precise** continuous node, or an `Interval`/`ProbabilityBox` for an **imprecise** 
continuous node [beer_imprecise_2013-1](@cite). Imprecision decides the resulting network type. Among the purely discrete networks, one
holding an imprecise discrete node is a [`CredalNetwork`](@ref) rather than a
[`BayesianNetwork`](@ref); and an [`EnhancedBayesianNetwork`](@ref) that contains *any*
imprecise node — discrete, continuous, or functional — reduces to a [`CredalNetwork`](@ref)
instead of a [`BayesianNetwork`](@ref).

## Nodes with an a-priori-known CPT

### Discrete nodes

A [`DiscreteNode`](@ref) holds a CPT over the combinations of its parents' states and its
own. A **root** is built from its name alone; a **child** additionally names its parents,
and every entry is filled with `node[parent => state, …, name => own_state] = p`. The
constructor enforces that the states are mutually exclusive and collectively exhaustive.

```@example nodes
using EnhancedBayesianNetworks # hide
W = DiscreteNode(:W)                        # root: CPT known a priori
W[:W => :sunny]  = 0.5
W[:W => :cloudy] = 0.5

S = DiscreteNode(:S, [:W])                  # child of W
S[:W => :sunny,  :S => :on] = 0.9; S[:W => :sunny,  :S => :off] = 0.1
S[:W => :cloudy, :S => :on] = 0.2; S[:W => :cloudy, :S => :off] = 0.8
S
```

A child's entries can be **mixed across scenarios**: some given as a `Real` probability and
others as an `Interval`. As soon as at least one entry is an `Interval`, the whole node is
**imprecise** — its CPT becomes a credal set — which calls for a [`CredalNetwork`](@ref)
rather than a [`BayesianNetwork`](@ref):

```@example nodes
S[:W => :sunny, :S => :on] = Interval(0.8, 0.95)
isprecise(S)
```

A discrete node can also carry per-state `parameters`. These are inert on their own; they
matter only when the node feeds a functional node, whose models then read them (see the
functional-nodes section below):

```@example nodes
P = DiscreteNode(:P, [:on => [Parameter(0.5, :P)], :off => [Parameter(0.0, :P)]])
P[:P => :on] = 0.7; P[:P => :off] = 0.3
P
```

### Continuous nodes

A [`ContinuousNode`](@ref) maps each parent-state combination to a continuous probability.
A **root** is built directly from a single distribution; a **child** names its parents and
assigns one distribution per parent-state combination:

```@example nodes
T = ContinuousNode(:T, Normal())            # root from one distribution
isroot(T), isprecise(T)
```

```@example nodes
C = ContinuousNode(:C, [:W])                # child: one distribution per parent state
C[:W => :sunny]  = Normal()
C[:W => :cloudy] = Normal(2, 1)
scenarios(C)
```

A continuous entry is precise when it is a `UnivariateDistribution`; using an `Interval` or
a `ProbabilityBox` [P_box_FAES](@cite) instead represents epistemic uncertainty. As with discrete nodes, a
child's entries may be mixed across parent-state combinations — a `UnivariateDistribution`
for some scenarios and an `Interval` or `ProbabilityBox` for others — and a single imprecise
entry makes the whole node imprecise.

### Discretization

Reduction eliminates continuous nodes, so a continuous node's posterior cannot be recovered
afterwards. To keep evidence observable on a continuous node — and to let it enter discrete
inference — it can be **discretized** [straub_bayesian_2010](@cite): its support is partitioned at a list of interval
edges into a discrete node plus a conditioned continuous remainder. The strategy is attached
to the node and depends on its position:

- a **root** carries an [`ExactDiscretization`](@ref) — the discrete probabilities follow
  *exactly* from the node's distribution, because the marginal is known;
- a **child** carries an [`ApproximatedDiscretization`](@ref) — the marginal is not
  generally available, so the tails are approximated (a uniform assumption over each
  bounded interval and an exponential assumption, with rate/spread `sigma`, over an
  unbounded tail).

```@example nodes
Tr = ContinuousNode(:Tr, Normal(), ExactDiscretization([-2.0, 0.0, 2.0]))   # root
Tr.discretization
```

```julia
# child: interval edges plus the exponential tail rate (here 1.5)
Cd = ContinuousNode(:Cd, [:W], ApproximatedDiscretization([-1.0, 0.0, 1.0], 1.5))
```

## Nodes with an a-priori-unknown CPT (functional nodes)

When a node's CPT is **not** known a priori, it is defined by a functional relationship
with its parents: one or more
[UncertaintyQuantification.jl](https://github.com/JuliaUQ/UncertaintyQuantification.jl)
models, plus a simulation that propagates the parents' uncertainty through them. The resulting table is a collection of
*structural reliability problems*, evaluated only when the enclosing network is
[`reduce`](@ref)d. A functional node is therefore always a child, and never a root.

A [`DiscreteFunctionalNode`](@ref) derives two states — `:<name>_safe` and
`:<name>_failed`. A `performance` function maps the models' output to a limit state
(*failed* where `performance < 0`), and the evaluated CPT stores the estimated failure
probability against `:<name>_failed` and its complement against `:<name>_safe`:

```@example functional
using EnhancedBayesianNetworks # hide
model = Model(df -> df.x .^ 2, :y)          # y is computed from parent x
performance = df -> df.y .- 1.0             # failed when y < 1

DF = DiscreteFunctionalNode(:DF, [model], performance, MonteCarlo(1000))
states(DF)                                  # [:DF_safe, :DF_failed]
```

A [`ContinuousFunctionalNode`](@ref) has no performance function: after evaluation, its
model output samples are fitted into an `EmpiricalDistribution`, one per scenario of its
discrete ancestors.

```julia
model = Model(df -> df.x .^ 2, :y)
CF = ContinuousFunctionalNode(:CF, [model], MonteCarlo(1000))
```

### One simulation, or one per scenario

Passing a single simulation — `MonteCarlo(1000)` above — reuses it for every scenario of
the node's discrete ancestors. To tune the effort, or even the method, per scenario, list
the parents explicitly in the constructor and then assign a simulation to each scenario the
same way you would fill a CPT:

```julia
DF = DiscreteFunctionalNode(:DF, [:a, :b], [model], performance)   # list the parents
DF[:a => :a1, :b => :b1] = MonteCarlo(1000)
DF[:a => :a1, :b => :b2] = SubSetSimulation(500, 0.1, 10, Uniform(-0.2, 0.2))
DF[:a => :a2, :b => :b1] = MonteCarlo(200)
DF[:a => :a2, :b => :b2] = MonteCarlo(200)
```

Different scenarios may use entirely different techniques (standard Monte Carlo, Subset
Simulation, a `DoubleLoop`, `RandomSlicing`, …). The same per-scenario form is available for
[`ContinuousFunctionalNode`](@ref) via its `ContinuousFunctionalNode(name, parents, models)`
constructor.

Constructing a functional node only records its models and simulation — no sampling runs
until reduction, which is why `states(DF)` above returns immediately. Like any discrete
node, a [`DiscreteFunctionalNode`](@ref) may carry `parameters` for when it in turn feeds a further
functional node.

!!! note "Imprecise parents and the double loop"
    A functional node with only precise parents can use any single-loop simulation (FORM,
    Monte Carlo, Line/Importance/Subset sampling). If **any** parent is imprecise (interval
    or p-box), the analysis needs a *double-loop* scheme — an outer optimization over the
    imprecise sets — and the reduced network becomes a [`CredalNetwork`](@ref).

## Inspecting nodes

The same accessors work across node types:

- [`states`](@ref) — the discrete states a node can take.
- [`scenarios`](@ref) — each CPT row as `parent => state` pairs.
- [`parents`](@ref) — the node's parent names (empty for a root).
- [`isroot`](@ref) — whether the node has no parents.
- [`isprecise`](@ref) — whether every entry is precise.
- [`sample`](@ref) — draw a state from a precise discrete node given evidence.

```@example nodes
(states = states(W), parents = parents(S), root = isroot(W))
```

Sampling draws a state consistent with fixed parent evidence (precise nodes only):

```julia
sample(S, Evidence(:W => :sunny))           # e.g. :on
```
