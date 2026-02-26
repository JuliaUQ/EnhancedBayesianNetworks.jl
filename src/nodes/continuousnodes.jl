struct ContinuousNode <: AbstractContinuousNode
    name::Symbol
    cpt::ConditionalProbabilityTable{ContinuousProbability}
    discretization::AbstractDiscretization
    results::Dict{Vector{Symbol},Tuple}

    function ContinuousNode(
        name::Symbol,
        parents::Vector{Symbol}=Symbol[],
        discretization::AbstractDiscretization=ExactDiscretization(),
        results::Dict{Vector{Symbol},Tuple}=Dict{Vector{Symbol},Tuple}()
    )
        if name == :Π
            error(":Π is not allowed as node name")
        end
        if isempty(discretization) && !isempty(parents)
            discretization = ApproximatedDiscretization()
        end
        if isempty(parents) && isa(discretization, ApproximatedDiscretization)
            error("Invalid eBN: node $name, is a root node and the discretization must be an ExactDiscretization")
        elseif !isempty(parents) && isa(discretization, ExactDiscretization)
            error("Invalid eBN: node $name, is a child node and the discretization must be an ApproximatedDiscretization")
        end

        cpt = ConditionalProbabilityTable{ContinuousProbability}(parents)
        return new(name, cpt, discretization, results)
    end
end

ContinuousNode(name::Symbol, discretization::AbstractDiscretization) = ContinuousNode(name, Symbol[], discretization, Dict{Vector{Symbol},Tuple}())

Base.setindex!(node::ContinuousNode, value, key...) = setindex!(node.cpt, value, key...)
Base.getindex(node::ContinuousNode, key...) = getindex(node.cpt, key...)

scenarios(node::ContinuousNode) = map(row -> [Symbol(col) => row[col] for col in names(node.cpt.data[:, Not("Π")])], eachrow(node.cpt.data[:, Not("Π")]))

isprecise(node::ContinuousNode) = all(isa.(node.cpt.data[:, :Π], UnivariateDistribution))

isroot(node::ContinuousNode) = size(node.cpt.data, 2) == 1

parents(node::ContinuousNode) = Symbol.(names(node.cpt.data[:, Not("Π")]))

function _inputs(node::ContinuousNode, evidence::Evidence)
    new_evidence = filter(((k, v),) -> k ∈ Symbol.(names(node.cpt.data)), evidence)
    dist = node.cpt[new_evidence...]
    if isa(dist, UnivariateDistribution) || isa(dist, ProbabilityBox)
        return RandomVariable(dist, node.name)
    elseif isa(dist, Interval)
        return IntervalVariable(dist.lb, dist.ub, node.name)
    end
end

function _distribution_bounds(node::ContinuousNode)
    bounds = mapreduce(dist -> _distribution_bounds(dist), hcat, node.cpt.data[!, :Π])
    return [minimum(bounds[1, :]), maximum(bounds[2, :])]
end

function _distribution_bounds(dist::UnivariateDistribution)
    return [support(dist).lb, support(dist).ub]
end

function _distribution_bounds(dist::Union{Interval,ProbabilityBox})
    return [dist.lb, dist.ub]
end

function _truncate(dist::UnivariateDistribution, bounds::Tuple{Real,Real})
    return truncated(dist, bounds[1], bounds[2])
end

function _truncate(_::Interval, bounds::Tuple{Real,Real})
    return Interval(bounds[1], bounds[2])
end

function _truncate(dist::ProbabilityBox, bounds::Tuple{Real,Real})
    return ProbabilityBox{typeof(dist).parameters[1]}(dist.parameters, bounds[1], bounds[2])
end