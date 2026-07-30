# Networks

A network is a directed acyclic graph whose vertices are [nodes](nodes.md) and whose
edges encode direct probabilistic dependence: an edge `parent → child` means the child's
distribution is conditioned on the parent. Three network types make up the modelling
front-end, differing only in the kind of nodes they admit:

| Type | Nodes | Purpose |
|:--|:--|:--|
| [`BayesianNetwork`](@ref) | discrete, **precise** | classical BN, ready for inference |
| [`CredalNetwork`](@ref) | discrete, some **imprecise** | credal network over interval CPTs |
| [`EnhancedBayesianNetwork`](@ref) | discrete + continuous + functional | general model that is *reduced* to one of the above |

The [`EnhancedBayesianNetwork`](@ref) (eBN) is the expressive modelling layer: it may mix
discrete, continuous, and functional nodes [straub_bayesian_2010](@cite). Inference is not
performed on it directly — it is first [`reduce`](@ref)d to a purely discrete
[`BayesianNetwork`](@ref) or [`CredalNetwork`](@ref), depending on whether any imprecision
survives. A [`BayesianNetwork`](@ref) or [`CredalNetwork`](@ref) can also be built directly
when the model is already discrete [jensen2007bayesian](@cite).

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
structure is complete, before reduction, inference, or sampling.

## Bayesian networks

A [`BayesianNetwork`](@ref) is a DAG of discrete, precise nodes — the classical formulation
[jensen2007bayesian](@cite). Its constructor rejects any imprecise node, pointing you to a
[`CredalNetwork`](@ref) instead, and it requires node names and states to be globally unique.
Once ordered, it supports the full inference and sampling machinery (see the Inference
chapter).

## Credal networks

When a discrete CPT carries interval-valued entries, the node is *imprecise* and the network
becomes a [`CredalNetwork`](@ref). Each local CPT is then a closed convex set of probability
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
- [`markov_envelope`](@ref) — the sub-networks used during reduction (below).

```@example networks
(parents = parents(ebn, :F), discrete_ancestors = discrete_ancestors(ebn, :F))
```

## Reduction

[`reduce`](@ref) turns an [`EnhancedBayesianNetwork`](@ref) into an inference-ready discrete
network. The transformation follows the enhanced-Bayesian-network procedure of
[straub_bayesian_2010](@cite) and rests on the node-removal operations for evaluating
influence diagrams [Shachter86a](@cite):

1. **Order** the network ([`order!`](@ref)).
2. **Discretize** every continuous node that carries a discretization strategy — it is
   replaced by a discrete surrogate (the per-interval probability masses) plus a residual
   continuous node, with parents rewired to the discrete part and children to the continuous
   part (`discretize!`).
3. **Transfer** each continuous-functional node's models into its children
   (`_transfer_continuous_functional_node!`). This is a computational optimization of the
   evaluation stage. A continuous-functional node that merely feeds another functional node
   would otherwise be evaluated on its own — sampling its models and fitting an
   `EmpiricalDistribution` per scenario — only for that distribution to be re-sampled by the
   child and then thrown away when the node is eliminated. Instead, its models are *prepended*
   to the child's model chain, so the samples already drawn for the child are propagated
   straight through them during the child's single structural reliability problem. The
   intermediate empirical distribution is never built for a node that reduction removes anyway.
4. **Evaluate** functional nodes in dependency order. A node whose parents are all
   non-functional has ready inputs: its conditional table is filled by solving a structural
   reliability problem over the scenario grid of its
   [`discrete_ancestors`](@ref), using the node's simulation
   [behrensdorf_uncertaintyquantificationjl_2023](@cite). The functional node is then replaced
   by the resulting discrete (or continuous) node.
5. **Eliminate** continuous parents. A continuous node that fed only the just-evaluated node
   is removed and its parents reconnected to its children (`_eliminate_node!` — node removal
   [Shachter86a](@cite)); one that still feeds other functional nodes keeps its remaining
   edges, and only the spent edge is cut.
6. **Dispatch** to the concrete type: once no node is continuous, an all-precise network
   becomes a [`BayesianNetwork`](@ref) and a network with any surviving imprecision a
   [`CredalNetwork`](@ref).

```julia
reduced = reduce(ebn)                       # -> BayesianNetwork (or CredalNetwork if imprecise)
```

Structural reliability problems are not solved over the whole network but on **Markov
envelopes** [straub_bayesian_2010](@cite): continuous nodes linked through their Markov
blankets are grouped, and each group's envelope — the union of its members' Markov blankets —
is the minimal sub-network on which the corresponding reliability problem is evaluated.
[`markov_envelope`](@ref) exposes these groups:

```@example networks
markov_envelope(ebn)
```

!!! note "Imprecision propagates to the reduced network"
    If any node feeding a structural reliability problem is imprecise — an interval-valued
    discrete parent, or a continuous parent given as an `Interval` or `ProbabilityBox` — the
    evaluated failure probability is itself an interval, obtained through a double-loop
    analysis. Reduction then yields a [`CredalNetwork`](@ref) rather than a
    [`BayesianNetwork`](@ref).
