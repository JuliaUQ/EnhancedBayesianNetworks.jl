function _discretize(node::ContinuousNode)
    intervals = _format_interval(node)
    name_discrete = Symbol(string(node.name) * "_d")
    discretized_node = DiscreteNode(name_discrete, parents(node))
    new_continuous = ContinuousNode(node.name, [name_discrete])
    if isroot(node)
        map(i -> discretized_node[(name_discrete .=> Symbol(i))] = _discretize(node[], i), intervals)
        map(i -> new_continuous[(name_discrete .=> Symbol(i))] = _truncate(node[], Tuple(i)), intervals)
    else
        for i in intervals
            map((sc) -> discretized_node[(vcat(first.(sc), name_discrete) .=> vcat(last.(sc), Symbol(i)))...] = _discretize(node[(sc)...], i), scenarios(node))
        end
        map(i -> new_continuous[(name_discrete .=> Symbol(i))] = _approximate(i, node.discretization.sigma), intervals)
    end
    return (discretized_node, new_continuous)
end

function _format_interval(node::ContinuousNode)
    intervals = Float64.(node.discretization.intervals)
    min = node.discretization.intervals[1]
    max = node.discretization.intervals[end]
    lower_bound, upper_bound = _distribution_bounds(node)
    if minimum(min) > lower_bound
        @warn "node $(repr(node.name)) has minimum intervals value $min > support lower bound $lower_bound. Lower bound will be used as intervals start"
        insert!(intervals, 1, lower_bound)
    end
    if minimum(min) < lower_bound
        @warn "node $(repr(node.name)) has minimum intervals value $min < support lower bound $lower_bound. Lower bound will be used as intervals start"
        deleteat!(intervals, intervals .<= lower_bound)
        insert!(intervals, 1, lower_bound)
    end
    if maximum(max) < upper_bound
        @warn "node $(repr(node.name)) has maximum intervals value $max < support upper bound $upper_bound. Upper bound will be used as intervals end"
        push!(intervals, upper_bound)
    end
    if maximum(max) > upper_bound
        @warn "node $(repr(node.name)) has maximum intervals value $max > support upper bound $upper_bound. Upper bound will be used as intervals end"
        deleteat!(intervals, intervals .>= upper_bound)
        push!(intervals, upper_bound)
    end
    return [[intervals[i], intervals[i+1]] for i in 1:(length(intervals)-1)]
end

function _approximate(i::AbstractVector{<:Real}, λ::Real)
    if all(isfinite.(i))
        return Uniform(i...)
    elseif isfinite(last(i))
        return -Exponential(λ) + last(i)
    elseif isfinite(first(i))
        return Exponential(λ) + first(i)
    end
end

function _discretize(dist::UnivariateDistribution, interval::Vector{<:Real})
    return cdf(dist, getindex(interval, 2)) - cdf(dist, getindex(interval, 1))
end

function _discretize(dist::ProbabilityBox, interval::Vector{<:Real})
    rb = cdf(dist, getindex(interval, 2))
    lb = cdf(dist, getindex(interval, 1))
    new_lb = minimum([rb.lb - lb.lb, rb.ub - lb.ub])
    new_ub = maximum([rb.lb - lb.lb, rb.ub - lb.ub])
    return Interval(new_lb, new_ub)
end

function _discretize(_::Interval, _::Vector{<:Real})
    return Interval(0, 1)
end