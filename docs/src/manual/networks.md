# Networks

A network is a directed acyclic graph (DAG) whose vertices are [nodes](nodes.md) and whose
edges encode direct probabilistic dependence: an edge `parent → child` means the child's
distribution is conditioned on the parent. Three network types make up the modelling
front-end, differing only in the kind of nodes they admit:

| Type | Nodes | Purpose |
|:--|:--|:--|
| [`BayesianNetwork`](@ref) | discrete, **precise** | classical BN, ready for inference |
| [`CredalNetwork`](@ref) | discrete, at least one **imprecise** | CN over interval CPTs, ready for inference |
| [`EnhancedBayesianNetwork`](@ref) | discrete + continuous + functional, **precise** or **imprecise** | general model that is *reduced* to one of the above |

The [`EnhancedBayesianNetwork`](@ref) (eBN) is the expressive modelling layer: it may mix
discrete, continuous, and functional nodes [straub_bayesian_2010](@cite). Inference is not
performed on it directly — it is first [`reduce`](@ref)d to a purely discrete
[`BayesianNetwork`](@ref) (BN) or [`CredalNetwork`](@ref) (CN), depending on whether any imprecision
survives. A [`BayesianNetwork`](@ref) [jensen2007bayesian](@cite) or
[`CredalNetwork`](@ref) [cozman_credal_2000](@cite) can also be built directly when the
model is already discrete.

## Building and validating a network

Every network is assembled the same way: construct the nodes, group them into a network,
wire the edges with [`add_child!`](@ref), and finalize with [`order!`](@ref).

```@example networks
using EnhancedBayesianNetworks # hide
W = DiscreteNode(:W)
W[:W => :sunny]  = 0.5
W[:W => :cloudy] = 0.5

S = DiscreteNode(:S, [:W])
S[:W => :sunny,  :S => :on] = 0.9; S[:W => :sunny,  :S => :off] = 0.1
S[:W => :cloudy, :S => :on] = 0.2; S[:W => :cloudy, :S => :off] = 0.8

bn = BayesianNetwork([W, S])
add_child!(bn, :W, :S)                      # wire parent → child (by name or by node)
order!(bn)                                  # topologically sort and validate
```

[`add_child!`](@ref) records directed edges. Each endpoint may be a single node or a vector,
given by name (`Symbol`) or as node objects, so a whole fan-out can be wired at once —
`add_child!(net, :W, [:S, :R])`. It rejects self-loops, requires every referenced node to
exist, checks that a discrete parent appears in each non-functional child's CPT, and enforces
the structural rule that **continuous and functional parents may feed only functional
children** (a continuous quantity cannot condition a plain discrete CPT).

[`order!`](@ref) sorts the nodes into a topological order and runs the global checks: the
graph must be acyclic and connected, no CPT may reference a parent that was never linked, and
every discrete CPT must be exhaustive over all parent/own-state combinations. Run it once the
structure is complete, before [reduction](reduction.md), [inference](inference.md), or sampling.

## Bayesian networks

A [`BayesianNetwork`](@ref) is a DAG of discrete, precise nodes — the classical formulation
[jensen2007bayesian](@cite). Its constructor rejects any imprecise node, pointing you to a
[`CredalNetwork`](@ref) instead, and it requires node names and states to be globally unique.
Once ordered, it supports the full inference and sampling machinery (see the
[Inference](inference.md) chapter).

## Credal networks

When a discrete CPT carries interval-valued entries, the node is *imprecise* and the network
becomes a [`CredalNetwork`](@ref) [cozman_credal_2000](@cite). Each local CPT is then a closed convex set of probability
measures — a *credal set* [Levi1980-LEVTEO-7](@cite) — and the network stands for the whole
family of Bayesian networks that share its graph but differ in the measures drawn from those
sets. Imprecision is expressed with interval probabilities [weichselberger_theory_2000](@cite),
or, for continuous quantities upstream, with probability boxes [P_box_FAES](@cite).

```@example networks
Wc = DiscreteNode(:Wc); Wc[:Wc => :sunny] = 0.5; Wc[:Wc => :cloudy] = 0.5
Sc = DiscreteNode(:Sc, [:Wc])
# a single interval entry makes the node — and hence the network — imprecise:
Sc[:Wc => :sunny,  :Sc => :on]  = Interval(0.8, 0.95); Sc[:Wc => :sunny,  :Sc => :off] = Interval(0.05, 0.2)
Sc[:Wc => :cloudy, :Sc => :on]  = 0.2;                 Sc[:Wc => :cloudy, :Sc => :off] = 0.8

cn = CredalNetwork([Wc, Sc])
add_child!(cn, :Wc, :Sc); order!(cn)
```

Constructing a [`CredalNetwork`](@ref) whose nodes all turn out precise emits a warning — a
[`BayesianNetwork`](@ref) is the right structure in that case. The reverse transition happens
automatically: after reduction, a credal network whose imprecision has vanished is narrowed
back to a [`BayesianNetwork`](@ref).

## Enhanced Bayesian networks

An [`EnhancedBayesianNetwork`](@ref) is the general modelling front-end
[straub_bayesian_2010](@cite): it may hold discrete nodes, continuous nodes, and functional
nodes side by side. Continuous and functional nodes carry the physics of the problem — the
[UncertaintyQuantification.jl](https://github.com/JuliaUQ/UncertaintyQuantification.jl) models
and simulations that define a functional node's conditional table
[behrensdorf_uncertaintyquantificationjl_2023](@cite) — so the eBN cannot be queried directly.
It is instead transformed into a discrete network by [`reduce`](@ref).

```@example networks
Wf = DiscreteNode(:Wf, [:sunny => [Parameter(1.0, :Wf)], :cloudy => [Parameter(2.0, :Wf)]])
Wf[:Wf => :sunny] = 0.5; Wf[:Wf => :cloudy] = 0.5
X = ContinuousNode(:X, Uniform(-1, 1), ExactDiscretization([-1.0, 0.0, 1.0]))
model = Model(df -> df.X .+ df.Wf, :Y)
F = DiscreteFunctionalNode(:F, [model], df -> df.Y, MonteCarlo(200))

ebn = EnhancedBayesianNetwork([Wf, X, F])
add_child!(ebn, :Wf, :F); add_child!(ebn, :X, :F); order!(ebn)
```

## Inspecting structure

Once ordered, a network can be queried for its local structure. These accessors take the
network and a node name:

- [`parents`](@ref) / [`children`](@ref) — the direct predecessors / successors of a node.
- [`discrete_ancestors`](@ref) — the discrete nodes reachable upstream, skipping continuous
  ones; these define the scenario grid over which a functional node is evaluated.
- [`markov_blanket`](@ref) — a node's parents, children, and co-parents; the minimal set that
  renders it conditionally independent of the rest of the network.
- [`markov_envelope`](@ref) — the groups of continuous nodes linked through shared Markov
  blankets, together with those blankets; a structural query over the network.

Formally, the Markov blanket of a node ``X_i`` is the union of its parents ``\mathrm{Pa}(X_i)``,
its children ``\mathrm{Ch}(X_i)``, and its *spouses* ``\mathrm{Sp}(X_i)`` — the other parents of
its children [jensen2007bayesian](@cite):

```math
\mathrm{Bl}(X_i) = \mathrm{Pa}(X_i) \cup \mathrm{Ch}(X_i) \cup \mathrm{Sp}(X_i).
```

```@example networks
(parents = parents(ebn, :F), discrete_ancestors = discrete_ancestors(ebn, :F))
```

```@example networks
markov_envelope(ebn)
```

## Reduction

An [`EnhancedBayesianNetwork`](@ref) is not queried directly — it is transformed into an
inference-ready discrete network by [`reduce`](@ref), which discretizes its continuous nodes and
evaluates its functional nodes as structural reliability problems, yielding a
[`BayesianNetwork`](@ref) or a [`CredalNetwork`](@ref). Because reduction is the core operation
of the library — and the route through which imprecision reaches the inference result — it has
its own chapter: [Reduction & Reliability Analysis](reduction.md).
