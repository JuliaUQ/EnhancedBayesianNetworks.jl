# Nodes

Nodes are the fundamental building blocks of a [Networks](@ref): together with the edges, they graphically represent the random variables of the Network. 
Every node carries a name (a `Symbol`) that uniquely identifies it, and *Conditional Probability Table* (CPT), which might by a priori **known** or **unknown**, and **continuous** or **discrete**. 
A node is classified along two independent lines.

**By position in the graph** a node is either a **root**: no parents, its state depending on nothing else. Or a **child**: with one or more parents that influence it. 
This is a property of the assembled network, read back with [`isroot`](@ref), not a separate type.

**By the nature of its variable** a node is either **discrete**: a finite set of mutually
exclusive states (discrete CPT). Or **continuous**: a real-valued quantity described by a probability distribution (continuous CPT).

Cutting across both is the distinction that decides *which type you actually construct*:

> **Is the node's conditional probability table (CPT) known a priori?**
>
> - **Yes**: you supply it explicitly, as a [`DiscreteNode`](@ref) or a
>   [`ContinuousNode`](@ref).
> - **No**: the CPT is instead defined by a *functional relationship* with the parents,
>   so the node **must be functional** ([`DiscreteFunctionalNode`](@ref),
>   [`ContinuousFunctionalNode`](@ref)). Its table does not exist up front: it is
>   materialized only when the enhanced Bayesian Network is reduced, by propagating the parents' uncertainty
>   through the node's models ([structural reliability problems](reduction.md)).

| Type | Variable | CPT | Role |
|:--|:--|:--|:--|
| [`DiscreteNode`](@ref) | discrete | known a priori | root or child |
| [`ContinuousNode`](@ref) | continuous | known a priori | root or child |
| [`DiscreteFunctionalNode`](@ref) | discrete | **not** known a priori | child only |
| [`ContinuousFunctionalNode`](@ref) | continuous | **not** known a priori | child only |

The distributions, models and simulation methods a node is built from belong to [UncertaintyQuantification.jl](https://github.com/JuliaUQ/UncertaintyQuantification.jl) [behrensdorf_uncertaintyquantificationjl_2023](@cite) and are re-exported by EnhancedBayesianNetworks, so they are available without a separate `using`.

CPT's entries may also be **precise** or **imprecise**: a discrete entry is a `Real` probability for a **precise** discrete node or a *probability* [`Interval`](@extref `UncertaintyQuantification.Interval`) for an **imprecise** (or **credal**) discrete node [weichselberger_theory_2000, Levi1980-LEVTEO-7](@cite); 
a continuous entry is a `UnivariateDistribution` for a **precise** continuous node, while it is an [`Interval`](@extref `UncertaintyQuantification.Interval`) or[`ProbabilityBox`](@extref `UncertaintyQuantification.ProbabilityBox`) [P_box_FAES](@cite) for an **imprecise** continuous node [beer_imprecise_2013-1](@cite). 
Imprecision decides the resulting network type. 

Among the purely discrete networks, one holding an imprecise discrete node is a [Credal Network](@ref) (CN) rather than a [Bayesian Network](@ref) (BN).
An [Enhanced Bayesian Network](@ref) (eBN) that contains *any* imprecise node reduces to a CN instead of a BN.

## Nodes with an a-priori-known CPT

### Discrete nodes

A [`DiscreteNode`](@ref) holds a CPT over the combinations of its parents' states and its own. 
A **root** is built from its name alone; a **child** additionally names its parents, and every entry is filled with `node[parent => state, …, name => own_state] = p`. 
The constructor enforces that the states are mutually exclusive and collectively exhaustive.

```@example nodes
using EnhancedBayesianNetworks # hide
W = DiscreteNode(:W)                        # root: CPT known a priori
W[:W => :sunny]  = 0.5
W[:W => :cloudy] = 0.5

S = DiscreteNode(:S, [:W])                  # child of W
S[:W => :sunny,  :S => :on] = 0.9 
S[:W => :sunny,  :S => :off] = 0.1
S[:W => :cloudy, :S => :on] = 0.2 
S[:W => :cloudy, :S => :off] = 0.8
S
```

A child's entries can be **mixed across scenarios**: some given as a `Real` probability and
others as an [`Interval`](@extref `UncertaintyQuantification.Interval`). 
As soon as at least one entry is an Interval, the whole node is **imprecise**, its CPT becomes a credal set, which calls for a CN rather than a BN:

```@example nodes
S[:W => :sunny, :S => :on] = Interval(0.8, 0.95)
S
```
```@example nodes
isprecise(S)
```

A discrete node can also carry per-state [`Parameter`](@extref `UncertaintyQuantification.Parameter`)s. 
These are inert on their own; they matter only when the node feeds a functional node, whose models ([`UQModel`](@extref `UncertaintyQuantification.UQModel`)s) read them (see the functional-nodes section below):

```@example nodes
P = DiscreteNode(:P, [:on => [Parameter(0.5, :P)], :off => [Parameter(0.0, :P)]])
P[:P => :on] = 0.7
P[:P => :off] = 0.3
P
```

### Continuous nodes

A [`ContinuousNode`](@ref) maps each parent-state combination to a probability distribution.
A **root** is built directly from a single distribution; a **child** names its parents and assigns one distribution per parent-state combination:

```@example nodes
T = ContinuousNode(:T, Normal())            # root from one distribution
T
```
```@example nodes
isroot(T), isprecise(T)
```

```@example nodes
C = ContinuousNode(:C, [:W])                # child: one distribution per parent state
C[:W => :sunny]  = Normal()
C[:W => :cloudy] = Normal(2, 1)
C
```
```@example nodes
scenarios(C)
```

A continuous entry is **precise** when it is a `UnivariateDistribution`; using an [`Interval`](@extref `UncertaintyQuantification.Interval`) or a [`ProbabilityBox`](@extref `UncertaintyQuantification.ProbabilityBox`) instead represents epistemic uncertainty (i.e. **imprecise** entry). 
As with discrete nodes, a child's entries may be **mixed across parent-state combinations**, a `UnivariateDistribution` for some scenarios and an Interval` or a Probability Box for others, and a single imprecise entry makes the whole node **imprecise**.

#### Discretization

[Reduction](reduction.md) eliminates continuous nodes, so a continuous node's posterior cannot be recovered afterward. 
To keep evidence observable on a continuous node and to let it enter discrete inference, it can be **discretized** [straub_bayesian_2010](@cite): its support is partitioned at a list of interval edges into a discrete node plus a conditioned continuous remainder. 
The strategy is attached to the node and depends on its position:

- a **root** carries an [`ExactDiscretization`](@ref): the discrete probabilities follow *exactly* from the node's distribution, because the marginal is known;
- a **child** carries an [`ApproximatedDiscretization`](@ref): the marginal is not generally available, so the tails are approximated (a uniform assumption over each bounded interval and an exponential assumption, with rate/spread `sigma`, over an unbounded tail).

Writing ``x_{ik}^-`` and ``x_{ik}^+`` for the lower and upper edges of the ``k``-th discretization interval, the conditional CDF of the continuous remainder ``X_i'`` given the discrete state ``k`` follows one of three forms.
 
A discretization on a **root** truncates the node's own CDF ``F_{X_i}`` to the interval, exactly [straub_bayesian_2010](@cite):

```math
F_{X_i'}(x_i \mid k) =
\begin{cases}
0 & x_i \le x_{ik}^-, \\
\dfrac{F_{X_i}(x_i) - F_{X_i}(x_{ik}^-)}{F_{X_i}(x_{ik}^+) - F_{X_i}(x_{ik}^-)} & x_{ik}^- \le x_i \le x_{ik}^+, \\
1 & x_i \ge x_{ik}^+.
\end{cases}
```

While a discretization on a **child** cannot use the (unknown) marginal, so each bounded interval is filled with a **uniform** assumption,

```math
F_{X_i}(x_i \mid k) =
\begin{cases}
0 & x_i \le x_{ik}^-, \\
\dfrac{x_i - x_{ik}^-}{x_{ik}^+ - x_{ik}^-} & x_{ik}^- \le x_i \le x_{ik}^+, \\
1 & x_i \ge x_{ik}^+,
\end{cases}
```

and an unbounded (right) tail with an **exponential** assumption of rate ``\lambda`` (the `sigma`
argument):

```math
F_{X_i}(x_i \mid k) =
\begin{cases}
0 & x_i \le x_{ik}^-, \\
1 - \exp\!\left[-\lambda\,(x_i - x_{ik}^-)\right] & x_i > x_{ik}^-.
\end{cases}
```

```@example nodes
Tr = ContinuousNode(:Tr, Normal(), ExactDiscretization([-2.0, 0.0, 2.0]))   # root
```

```@example nodes
# child: interval edges plus the exponential tail rate (here 1.5)
Cd = ContinuousNode(:Cd, [:W], ApproximatedDiscretization([-1.0, 0.0, 1.0], 1.5))
```

## Nodes with an a-priori-unknown CPT (functional nodes)

When a node's CPT is **not** known a priori, it is defined by a functional relationship with its parents: one or more [`UQModel`](@extref `UncertaintyQuantification.UQModel`), plus a simulation that propagates the parents' uncertainty through them. 
Models range from simple analytical expression to `ExternalModel` wrapping any external solver, and among the simulation techniques we have standard `MonteCarlo`, [`FORM`](@extref `UncertaintyQuantification.FORM`), advanced Monte Carlo (e.g. `LineSampling`, `ImportantSampling`, [`SubSetSimulation`](@extref `UncertaintyQuantification.SubSetSimulation`)), [`DoubleLoop`](@extref `UncertaintyQuantification.DoubleLoop`) and [`RandomSlicing`](@extref `UncertaintyQuantification.RandomSlicing`). 
For a full description of each, see the [UncertaintyQuantification.jl](https://github.com/JuliaUQ/UncertaintyQuantification.jl) manual.

The resulting table is a collection of *structural reliability problems*, evaluated only when the network is reduced (see [Reduction & Structural Reliability Problem](reduction.md)). 
A functional node is therefore always a child, and never a root.

### Discrete Functional nodes

A [`DiscreteFunctionalNode`](@ref) derives two states: `:<name>_safe` and `:<name>_failed`. 
A `performance` function maps the models' output to a limit state (*failed* where `performance < 0`), and the evaluated CPT stores the estimated failure probability against `:<name>_failed` and its complement against `:<name>_safe`:

```@example nodes
model = Model(df -> df.x .^ 2, :y)          # y is computed from parent x
performance = df -> df.y .- 1.0             # failed when y < 1

DF = DiscreteFunctionalNode(:DF, [model], performance, MonteCarlo(1000))
```
```@example nodes
states(DF)                                  # [:DF_safe, :DF_failed]
```

### Continuous Functional nodes

A [`ContinuousFunctionalNode`](@ref) has no performance function: after evaluation, its model output samples are fitted into an [`EmpiricalDistribution`](@extref `UncertaintyQuantification.EmpiricalDistribution`), one per scenario of its discrete ancestors.

```@example nodes
model = Model(df -> df.x .^ 2, :y)
CF = ContinuousFunctionalNode(:CF, [model], MonteCarlo(1000))
```

#### One simulation, or one per scenario

Passing a single simulation, e.g `MonteCarlo(1000)` above, reuses it for every scenario of the node's discrete ancestors. 
To tune the effort, or even the method, per scenario, list the parents explicitly in the constructor and then assign a simulation to each scenario the same way you would fill a CPT:

```@example nodes
DF = DiscreteFunctionalNode(:DF, [:a, :b], [model], performance)   # list the parents
DF[:a => :a1, :b => :b1] = MonteCarlo(1000)
DF[:a => :a1, :b => :b2] = SubSetSimulation(500, 0.1, 10, Uniform(-0.2, 0.2))
DF[:a => :a2, :b => :b1] = MonteCarlo(200)
DF[:a => :a2, :b => :b2] = MonteCarlo(200)
DF
```

Different scenarios may use entirely different techniques (standard `MonteCarlo`, `SubSetSimulation`, a `DoubleLoop`, `RandomSlicing` or others). 
The same per-scenario form is available for [`ContinuousFunctionalNode`](@ref) via its `ContinuousFunctionalNode(name, parents, models)` constructor.

Constructing a functional node only records its models and simulation, no sampling runs until reduction, which is why `states(DF)` above returns immediately. 
Like any discrete node, a [`DiscreteFunctionalNode`](@ref) may carry `parameters` for when it in turn feeds a further functional node.

!!! note "**Imprecise parents**"
    A functional node with only *precise* parents can use any *single-loop simulation* (FORM, Monte Carlo, Advanced Monte Carlo). 
    If **any** parent is *imprecise*, the analysis needs a *double-loop simulation* (Double Loop or Random Slicing) and the reduced network becomes a CN.

## Inspecting nodes

The same accessors work across node types:

- [`states`](@ref): the discrete states a node can take.
- [`scenarios`](@ref): each CPT row as `parent => state` pairs.
- [`parents`](@ref): the node's parent names (empty for a root).
- [`isroot`](@ref): whether the node has no parents.
- [`isprecise`](@ref): whether every entry is precise.
- [`sample`](@ref): draw a state from a precise discrete node given evidence.

```@example nodes
(states = states(W), parents = parents(S), root = isroot(W))
```

Sampling draws a state consistent with fixed parent evidence (precise nodes only):

```@example nodes
S = DiscreteNode(:S, [:W])                  # child of W
S[:W => :sunny,  :S => :on] = 0.9
S[:W => :sunny,  :S => :off] = 0.1
S[:W => :cloudy, :S => :on] = 0.2
S[:W => :cloudy, :S => :off] = 0.8
sample(S, Evidence(:W => :sunny))           # e.g. :on
```