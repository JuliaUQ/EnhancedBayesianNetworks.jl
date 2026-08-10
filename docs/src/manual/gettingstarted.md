# Getting Started

## Installation

EnhancedBayesianNetworks.jl is not yet registered in Julia's General registry, so install it directly from its GitHub repository. From the Julia REPL, enter the package manager with `]` and add it by URL:

```julia
pkg> add https://github.com/JuliaUQ/EnhancedBayesianNetworks.jl
```

or, equivalently, from code:

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaUQ/EnhancedBayesianNetworks.jl")
```

Then load it — this also brings in the re-exported [UncertaintyQuantification.jl](https://github.com/JuliaUQ/UncertaintyQuantification.jl) types (`Model`, `Parameter`, `RandomVariable`, `Interval`, `ProbabilityBox`, the simulation methods)

```@example gettingstarted
using EnhancedBayesianNetworks
```

## Your first Bayesian network

Building a network always follows the same three steps: **construct the nodes**, **wire the edges** with [`add_child!`](@ref), and **finalize** with [`order!`](@ref).  
Here is a two-node weather/sprinkler model — a *root node* `W` and a *child node* `S` whose *Conditional Probability Table* (CPT) is conditioned on it:

```@example gettingstarted
W = DiscreteNode(:W)
W[:W => :sunny]  = 0.5
W[:W => :cloudy] = 0.5

S = DiscreteNode(:S, [:W])
S[:W => :sunny,  :S => :on] = 0.9; S[:W => :sunny,  :S => :off] = 0.1
S[:W => :cloudy, :S => :on] = 0.2; S[:W => :cloudy, :S => :off] = 0.8

bn = BayesianNetwork([W, S])
add_child!(bn, :W, :S)
order!(bn)
```

Now query it. [`infer`](@ref) returns the *posterior* over a query variable given some evidence:

```@example gettingstarted
infer(bn, :S, Evidence(:W => :sunny))       # P(S | W = sunny)
```

With no evidence you get the posterior is just the *prior* marginal:

```@example gettingstarted
infer(bn, :S, Evidence())                   # P(S)
```

## A first enhanced Bayesian network

The real power of the package is mixing in [`ContinuousNode`](@ref)s and *functional nodes* — a node whose CPT comes from a reliability analysis rather than being tabulated and can be either a [`DiscreteFunctionalNode`](@ref) or a [`ContinuousFunctionalNode`](@ref). 
The [`EnhancedBayesianNetwork`](@ref) (eBN) is reduced to a standard [`BayesianNetwork`](@ref) (BN) with [`reduce`](@ref) function, then queried exactly as above:

```julia
Load = DiscreteNode(:Load, [:low => [Parameter(1.0, :Load)], :high => [Parameter(3.0, :Load)]])
Load[:Load => :low] = 0.7; Load[:Load => :high] = 0.3
R = ContinuousNode(:R, Normal(3.0, 0.5))                 # a continuous resistance
model = Model(df -> df.R .- df.Load, :g)                 # limit state g = R - Load
F = DiscreteFunctionalNode(:F, [model], df -> df.g, MonteCarlo(2000))

ebn = EnhancedBayesianNetwork([Load, R, F])
add_child!(ebn, :Load, :F); add_child!(ebn, :R, :F); order!(ebn)

reduced = reduce(ebn)                                    # -> BayesianNetwork
infer(reduced, :F, Evidence(:Load => :high))            # failure probability given a high load
```

The package allows for an imprecise description of both [`ContinuousNode`](@ref)s and [`DiscreteNode`](@ref)s.
See [Reduction & Reliability Analysis](reduction.md) for the full story.

## Where to next

- [Introduction](introduction.md) — the concepts behind enhanced Bayesian networks.
- [Nodes](nodes.md) and [Networks](networks.md) — the building blocks in depth.
- [Reduction & Reliability Analysis](reduction.md) — evaluating enhanced networks, with and
  without imprecision.
- [Inference](inference.md), [Parameter Learning](parameterlearning.md), and
  [Plotting](plotting.md).
