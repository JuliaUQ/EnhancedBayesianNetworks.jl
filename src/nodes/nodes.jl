abstract type AbstractNode end
abstract type AbstractContinuousNode <: AbstractNode end
abstract type AbstractDiscreteNode <: AbstractNode end

abstract type AbstractDiscretization end

""" ExactDiscretization

    Used for ContinuousRootNode whenever evidence can be available on them.
        intervals: vector of Float64 that discretize initial distribution support

"""
struct ExactDiscretization <: AbstractDiscretization
    intervals::Vector{<:Real}

    function ExactDiscretization(intervals::Vector{<:Real})
        if !issorted(intervals)
            error("interval values $intervals are not sorted")
        end
        new(intervals)
    end
end

ExactDiscretization() = ExactDiscretization(Vector{Real}())

""" ApproximatedDiscretization

    Used for continuous Non-Root nodes whenever evidence can be available on them.
        intervals: vector of Float64 that discretize initial distribution support
        sigma: variance of the normal distribution used for appriximate initial continuous distribution

"""
struct ApproximatedDiscretization <: AbstractDiscretization
    intervals::Vector{<:Real}
    sigma::Real

    function ApproximatedDiscretization(intervals::Vector{<:Real}, sigma::Real)
        if !issorted(intervals)
            error("interval values $intervals are not sorted")
        elseif sigma < 0
            error("variance must be positive")
        elseif sigma > 2
            @warn "Selected variance values $sigma can be too big, and the approximation not realistic"
        end
        new(intervals, sigma)
    end
end

Base.isempty(d::AbstractDiscretization) = isempty(d.intervals)

ApproximatedDiscretization() = ApproximatedDiscretization(Vector{Real}(), 0)

struct ResultTable
    data::DataFrame
    function ResultTable(columns::Union{Symbol,Vector{Symbol}})
        columns = wrap(columns)
        data = DataFrame([col => Symbol[] for col in columns])
        data[:, :res] = []
        new(data)
    end
end

function Base.setindex!(rt::ResultTable, value, key::Pair{Symbol,Symbol}...)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    evidence_nodes = collect(map(p -> p[1], key))
    rt_nodes = Symbol.(filter(i -> i != "res", names(rt.data)))
    if issetequal(evidence_nodes, rt_nodes)
        cp = subset(rt.data, selector, view=true)
        if isempty(cp)
            push!(rt.data, (key..., res=value))
        else
            @assert size(cp, 1) == 1
            cp.res[1] = value
        end
    else
        error("Cannot set index with $evidence_nodes into a ResultTable initialized with $rt_nodes")
    end
end

function Base.getindex(rt::ResultTable, key::Pair{Symbol,Symbol}...)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    cp = subset(rt.data, selector, view=true)
    if isempty(cp)
        error("index not find in the SimlationTable $st")
    else
        @assert size(cp, 1) == 1
        return cp.res[1]
    end
end

function Base.filter(st::ResultTable, key::Pair{Symbol,Symbol}...)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    return subset(st.data, selector, view=true)
end

include("../utils/wrap.jl")
include("../utils/flat.jl")
include("cpts.jl")
include("st.jl")
include("discretenodes.jl")
include("continuousnodes.jl")
include("functionalnodes.jl")