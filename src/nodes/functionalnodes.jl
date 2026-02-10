struct ContinuousFunctionalNode <: AbstractNode
    name::Symbol
    models::AbstractVector{<:UQModel}
    simulation::AbstractMonteCarlo
    discretization::ApproximatedDiscretization

    function ContinuousFunctionalNode(
        name::Symbol,
        models::Union{Vector{<:UQModel},<:UQModel},
        simulation::AbstractMonteCarlo,
        discretization::ApproximatedDiscretization=ApproximatedDiscretization()
    )
        if name == :Π
            error(":Π is not allowed as node name")
        end
        new(name, wrap(models), simulation, discretization)
    end
end

function isa_generalized_continuous(n::AbstractNode)
    if isa(n, ContinuousNode)
        return true
    elseif isa(n, ContinuousFunctionalNode)
        return true
    else
        return false
    end
end

struct DiscreteFunctionalNode <: AbstractDiscreteNode
    name::Symbol
    models::AbstractVector{<:UQModel}
    performance::Function
    simulation::Union{AbstractSimulation,DoubleLoop,RandomSlicing}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteFunctionalNode(
        name::Symbol,
        models::Union{Vector{<:UQModel},<:UQModel},
        performance::Function,
        simulation::Union{AbstractSimulation,DoubleLoop,RandomSlicing},
        parameters::Dict{Symbol,Vector{Parameter}}=Dict{Symbol,Vector{Parameter}}()
    )
        if name == :Π
            error(":Π is not allowed as node name")
        end
        new(name, wrap(models), performance, simulation, parameters)
    end
end

function isa_generalized_discrete(n::AbstractNode)
    if isa(n, DiscreteNode)
        return true
    elseif isa(n, DiscreteFunctionalNode)
        return true
    else
        return false
    end
end

isroot(FunctionalNode) = false

const global FunctionalNode = Union{DiscreteFunctionalNode,ContinuousFunctionalNode}