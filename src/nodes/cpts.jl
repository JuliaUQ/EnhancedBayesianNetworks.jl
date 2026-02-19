const ContinuousProbability = Union{UnivariateDistribution,ProbabilityBox,Interval}
const DiscreteProbability = Union{Real,Interval}
const Probability = Union{ContinuousProbability,DiscreteProbability}

struct ConditionalProbabilityTable{T<:Union{ContinuousProbability,DiscreteProbability}}
    data::DataFrame
    function ConditionalProbabilityTable{T}(columns::Union{Symbol,Vector{Symbol}}) where {T<:Union{ContinuousProbability,DiscreteProbability}}
        columns = wrap(columns)
        data = DataFrame([col => Symbol[] for col in columns])
        data[:, :Π] = T[]
        return new{T}(data)
    end
end

function Base.setindex!(cpt::ConditionalProbabilityTable{DiscreteProbability}, value, key...)
    verify_probability_value(value)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    evidence_nodes = collect(map(p -> p[1], key))
    cpt_nodes = Symbol.(filter(i -> i != "Π", names(cpt.data)))
    if issetequal(evidence_nodes, cpt_nodes)
        cp = subset(cpt.data, selector, view=true)
        if isempty(cp)
            push!(cpt.data, (key..., Π=value))
        else
            @assert size(cp, 1) == 1
            cp.Π[1] = value
        end
    else
        error("Cannot set index with $evidence_nodes into a CPT initialized with $cpt_nodes")
    end
    return nothing
end

function Base.setindex!(cpt::ConditionalProbabilityTable{ContinuousProbability}, value, key...)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    evidence_nodes = collect(map(p -> p[1], key))
    cpt_nodes = Symbol.(filter(i -> i != "Π", names(cpt.data)))
    if issetequal(evidence_nodes, cpt_nodes)
        cp = subset(cpt.data, selector, view=true)
        if isempty(cp)
            push!(cpt.data, (key..., Π=value))
        else
            @assert size(cp, 1) == 1
            cp.Π[1] = value
        end
    else
        error("Cannot set index with $evidence_nodes into a CPT initialized with $cpt_nodes")
    end
    return nothing
end

function Base.getindex(cpt::ConditionalProbabilityTable, key...)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    cp = subset(cpt.data, selector, view=true)
    if isempty(cp)
        error("index $(collect(key)) not found in the CPT $cpt")
    else
        @assert size(cp, 1) == 1
        return cp.Π[1]
    end
end

function Base.filter(cpt::ConditionalProbabilityTable, key...)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    return subset(cpt.data, selector, view=true)
end

function verify_probability_value(value::Real)
    if !(0 <= value <= 1)
        throw(ArgumentError("probability $value must be >= 0 and <= 1"))
    end
    return nothing
end

function verify_probability_value(value::Interval)
    if !all(0 .<= UncertaintyQuantification.bounds(value) .<= 1)
        throw(ArgumentError("probability $value must be >= 0 and <= 1"))
    end
    return nothing
end

function verify_probability_value(_::UnivariateDistribution)
    return nothing
end

function verify_probability_value(_::ProbabilityBox)
    return nothing
end