"""
    ContinuousFunctionalNode(name, models, simulation, discretization=ApproximatedDiscretization(), nbins=0)
    ContinuousFunctionalNode(name, ancestors::Vector{Symbol}, models, discretization=ApproximatedDiscretization(), nbins=0)

A continuous node whose value is **computed** from its ancestors by one or more UncertaintyQuantification
`models`, instead of being stored as a table. When the node is evaluated, `simulation` (e.g.
`MonteCarlo`) propagates the parents' uncertainty through the models to produce output samples;
these samples are then turned back into a continuous distribution by fitting an
[`EmpiricalDistribution`](https://juliauq.github.io/UncertaintyQuantification.jl/stable/manual/kde)
with `nbins` bins. Separately, `discretization` (an `ApproximatedDiscretization`) holds the interval
edges used to discretize that continuous node into discrete states for downstream discrete inference.
The node is evaluated once per combination of its discrete ancestors, so its uncertainty comes only
from its direct parents while the discrete ancestors form the scenario grid the simulation is repeated
over. A functional node is never a root; its parents are the inputs referenced by the models, or may be
listed explicitly in the second form.

# Examples
```julia
model = Model(df -> df.x .^ 2, :y)                 # y is computed from parent x

# propagate x's uncertainty through the model with a Monte Carlo simulation:
CF = ContinuousFunctionalNode(:CF, [model], MonteCarlo(1000))

# use 10 bins when refitting the output's EmpiricalDistribution from the samples:
CFb = ContinuousFunctionalNode(:CF, [model], MonteCarlo(1000), 10)
```
"""
mutable struct ContinuousFunctionalNode <: AbstractContinuousNode
    name::Symbol
    models::AbstractVector{<:UQModel}
    simulation::Union{AbstractMonteCarlo,ScenariosTable{ContinuousSimulation}}
    discretization::ApproximatedDiscretization
    nbins::Int

    function ContinuousFunctionalNode(
        name::Symbol,
        models::Union{Vector{<:UQModel},<:UQModel},
        simulation::Union{AbstractMonteCarlo,ScenariosTable{ContinuousSimulation}},
        discretization::ApproximatedDiscretization=ApproximatedDiscretization(),
        nbins::Int=0,
    )
        if name == :Π
            error(":Π is not allowed as node name")
        end
        if name == :sim
            error(":sim is not allowed as node name")
        end
        new(name, wrap(models), simulation, discretization, nbins)
    end
end

function ContinuousFunctionalNode(
    name::Symbol,
    ancestors::Vector{Symbol},
    models::Union{Vector{<:UQModel},<:UQModel},
    discretization::ApproximatedDiscretization=ApproximatedDiscretization(),
    nbins::Int=0,
)
    ContinuousFunctionalNode(name, models, ScenariosTable{ContinuousSimulation}(ancestors, :sim), discretization, nbins)
end

function ContinuousFunctionalNode(
    name::Symbol,
    models::Union{Vector{<:UQModel},<:UQModel},
    simulation::Union{AbstractMonteCarlo,ScenariosTable{ContinuousSimulation}},
    nbins::Int
)
    ContinuousFunctionalNode(name, models, simulation, ApproximatedDiscretization(), nbins)
end

function ContinuousFunctionalNode(
    name::Symbol,
    ancestors::Vector{Symbol},
    models::Union{Vector{<:UQModel},<:UQModel},
    nbins::Int
)
    ContinuousFunctionalNode(name, ancestors, models, ApproximatedDiscretization(), nbins)
end

Base.setindex!(node::ContinuousFunctionalNode, value, key...) = Base.setindex!(node.simulation, value, key...)

"""
    DiscreteFunctionalNode(name, models, performance, simulation, parameters=[])
    DiscreteFunctionalNode(name, ancestors::Vector{Symbol}, models, performance, parameters=[])

A discrete node whose two states — `:<name>_safe` and `:<name>_failed` — come from a reliability
analysis rather than a table. When the node is evaluated, `simulation` (e.g. `MonteCarlo`) propagates
the parents' uncertainty through the `models`, and the `performance` function maps the models' output
to a limit state: the node is *failed* where `performance < 0` and *safe* otherwise. The estimated
failure probability is stored on `:<name>_failed` and its complement on `:<name>_safe`. The node is
evaluated once per combination of its discrete ancestors, so its uncertainty comes only from its direct
parents while the discrete ancestors form the scenario grid the simulation is repeated over. Optional
per-state `parameters` behave as in [`DiscreteNode`](@ref): they are forwarded to descendants when this
node feeds a further functional node. A functional node is never a root; its parents are the inputs
referenced by the models, or may be listed explicitly in the second form.

# Examples
```julia
model = Model(df -> df.x .^ 2, :y)
performance = df -> df.y .- 1.0                    # failed when y < 1

DF = DiscreteFunctionalNode(:DF, [model], performance, MonteCarlo(1000))
states(DF)                                         # [:DF_safe, :DF_failed]

# optional per-state parameters, keyed by the two derived states:
DFp = DiscreteFunctionalNode(:DF, [model], performance, MonteCarlo(1000),
    [:DF_safe => [Parameter(1.0, :DF)], :DF_failed => [Parameter(0.0, :DF)]])
```
"""
mutable struct DiscreteFunctionalNode <: AbstractDiscreteNode
    name::Symbol
    models::AbstractVector{<:UQModel}
    performance::Function
    simulation::Union{DiscreteSimulation,ScenariosTable{DiscreteSimulation}}
    parameters::Vector{Pair{Symbol,Vector{Parameter}}}

    function DiscreteFunctionalNode(
        name::Symbol,
        models::Union{Vector{<:UQModel},<:UQModel},
        performance::Function,
        simulation::Union{DiscreteSimulation,ScenariosTable{DiscreteSimulation}},
        parameters::Vector{Pair{Symbol,Vector{Parameter}}}=Vector{Pair{Symbol,Vector{Parameter}}}()
    )
        if name == :Π
            error(":Π is not allowed as node name")
        end
        if name == :sim
            error(":sim is not allowed as node name")
        end
        new(name, wrap(models), performance, simulation, parameters)
    end
end

function DiscreteFunctionalNode(
    name::Symbol,
    ancestors::Vector{Symbol},
    models::Union{Vector{<:UQModel},<:UQModel},
    performance::Function,
    parameters::Vector{Pair{Symbol,Vector{Parameter}}}=Vector{Pair{Symbol,Vector{Parameter}}}()
)
    st = ScenariosTable{DiscreteSimulation}(ancestors, :sim)
    DiscreteFunctionalNode(name, models, performance, st, parameters)
end

Base.setindex!(node::DiscreteFunctionalNode, value, key...) = Base.setindex!(node.simulation, value, key...)

states(node::DiscreteFunctionalNode) = Symbol.([string(node.name) * "_safe", string(node.name) * "_failed"])

const global FunctionalNode = Union{DiscreteFunctionalNode,ContinuousFunctionalNode}

isroot(node::FunctionalNode) = false
