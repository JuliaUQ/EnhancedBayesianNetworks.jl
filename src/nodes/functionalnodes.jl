mutable struct ContinuousFunctionalNode <: AbstractContinuousNode
    name::Symbol
    models::AbstractVector{<:UQModel}
    simulation::Union{AbstractMonteCarlo,SimulationTable{ContinuousSimulation}}
    discretization::ApproximatedDiscretization

    function ContinuousFunctionalNode(
        name::Symbol,
        models::Union{Vector{<:UQModel},<:UQModel},
        simulation::Union{AbstractMonteCarlo,SimulationTable{ContinuousSimulation}},
        discretization::ApproximatedDiscretization=ApproximatedDiscretization(),)
        if name == :Π
            error(":Π is not allowed as node name")
        end
        if name == :sim
            error(":sim is not allowed as node name")
        end
        new(name, wrap(models), simulation, discretization)
    end
end

function ContinuousFunctionalNode(
    name::Symbol,
    ancestors::Vector{Symbol},
    models::Union{Vector{<:UQModel},<:UQModel},
    discretization::ApproximatedDiscretization=ApproximatedDiscretization(),
)
    st = SimulationTable{ContinuousSimulation}(ancestors)
    return ContinuousFunctionalNode(name, models, st, discretization)
end

Base.setindex!(node::ContinuousFunctionalNode, value, key...) = setindex!(node.simulation, value, key...)

mutable struct DiscreteFunctionalNode <: AbstractDiscreteNode
    name::Symbol
    models::AbstractVector{<:UQModel}
    performance::Function
    simulation::Union{DiscreteSimulation,SimulationTable{DiscreteSimulation}}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteFunctionalNode(
        name::Symbol,
        models::Union{Vector{<:UQModel},<:UQModel},
        performance::Function,
        simulation::Union{DiscreteSimulation,SimulationTable{DiscreteSimulation}},
        parameters::Dict{Symbol,Vector{Parameter}}=Dict{Symbol,Vector{Parameter}}()
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
    parameters::Dict{Symbol,Vector{Parameter}}=Dict{Symbol,Vector{Parameter}}()
)
    st = SimulationTable{DiscreteSimulation}(ancestors)
    return DiscreteFunctionalNode(name, models, performance, st, parameters)
end

Base.setindex!(node::DiscreteFunctionalNode, value, key...) = setindex!(node.simulation, value, key...)

isroot(FunctionalNode) = false

states(node::DiscreteFunctionalNode) = Symbol.([string(node.name) * "_safe", string(node.name) * "_failed"])

const global FunctionalNode = Union{DiscreteFunctionalNode,ContinuousFunctionalNode}