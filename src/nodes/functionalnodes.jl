mutable struct ContinuousFunctionalNode <: AbstractContinuousNode
    name::Symbol
    models::AbstractVector{<:UQModel}
    simulation::Union{AbstractMonteCarlo,SimulationTable{ContinuousSimulation}}
    discretization::ApproximatedDiscretization
    nbins::Int

    function ContinuousFunctionalNode(
        name::Symbol,
        models::Union{Vector{<:UQModel},<:UQModel},
        simulation::Union{AbstractMonteCarlo,SimulationTable{ContinuousSimulation}},
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
    ContinuousFunctionalNode(name, models, SimulationTable{ContinuousSimulation}(ancestors), discretization, nbins)
end

function ContinuousFunctionalNode(
    name::Symbol,
    models::Union{Vector{<:UQModel},<:UQModel},
    simulation::Union{AbstractMonteCarlo,SimulationTable{ContinuousSimulation}},
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

mutable struct DiscreteFunctionalNode <: AbstractDiscreteNode
    name::Symbol
    models::AbstractVector{<:UQModel}
    performance::Function
    simulation::Union{DiscreteSimulation,SimulationTable{DiscreteSimulation}}
    parameters::Vector{Pair{Symbol,Vector{Parameter}}}

    function DiscreteFunctionalNode(
        name::Symbol,
        models::Union{Vector{<:UQModel},<:UQModel},
        performance::Function,
        simulation::Union{DiscreteSimulation,SimulationTable{DiscreteSimulation}},
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
    st = SimulationTable{DiscreteSimulation}(ancestors)
    DiscreteFunctionalNode(name, models, performance, st, parameters)
end

Base.setindex!(node::DiscreteFunctionalNode, value, key...) = Base.setindex!(node.simulation, value, key...)

isroot(FunctionalNode) = false

states(node::DiscreteFunctionalNode) = Symbol.([string(node.name) * "_safe", string(node.name) * "_failed"])

const global FunctionalNode = Union{DiscreteFunctionalNode,ContinuousFunctionalNode}