struct ContinuousFunctionalNode <: AbstractContinuousNode
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

isroot(FunctionalNode) = false

const global FunctionalNode = Union{DiscreteFunctionalNode,ContinuousFunctionalNode}