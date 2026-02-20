function _format_interval(node::ContinuousNode)
    intervals = node.discretization.intervals
    intervals = convert(Vector{Float64}, intervals)
    min = node.discretization.intervals[1]
    max = node.discretization.intervals[end]
    lower_bound, upper_bound = _distribution_bounds(node)
    if minimum(min) > lower_bound
        @warn "node $(node.name) has minimum intervals value $min > support lower bound $lower_bound. Lower bound will be used as intervals start"
        insert!(intervals, 1, lower_bound)
    end
    if minimum(min) < lower_bound
        @warn "node $(node.name) has minimum intervals value $min < support lower bound $lower_bound. Lower bound will be used as intervals start"
        deleteat!(intervals, intervals .<= lower_bound)
        insert!(intervals, 1, lower_bound)
    end
    if maximum(max) < upper_bound
        @warn "node $(node.name) has maximum intervals value $max < support upper bound $upper_bound. Upper bound will be used as intervals end"
        push!(intervals, upper_bound)
    end
    if maximum(max) > upper_bound
        @warn "node $(node.name) has maximum intervals value $max > support upper bound $upper_bound. Upper bound will be used as intervals end"
        deleteat!(intervals, intervals .>= upper_bound)
        push!(intervals, upper_bound)
    end
    return [[intervals[i], intervals[i+1]] for i in 1:length(intervals)-1]
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
    minimum([rb.lb - lb.lb, rb.ub - lb.ub]), maximum([rb.lb - lb.lb, rb.ub - lb.ub])
end

function _discretize(_::Interval, _::Vector{<:Real})
    return (0, 1)
end

function _discretize(node::ContinuousNode)
    intervals = _format_interval(node)
    name_discrete = Symbol(string(node.name) * "_d")
    discretized_node = DiscreteNode(name_discrete, parents(node))
    if isroot(node)
        map(i -> discretized_node[(name_discrete.=>Symbol(i))] = _discretize(node[], i), intervals)
    else
        for i in intervals
            map((sc) -> discretized_node[(vcat(first.(sc), name_discrete) .=> vcat(last.(sc), Symbol(i)))...] = _discretize(node[(sc)...], i), scenarios(node))
        end
    end
    return discretized_node
end

# function _discretize!(net::EnhancedBayesianNetwork)
#     continuous_nodes = filter(x -> isa(x, ContinuousNode), net.nodes)
#     evidence_nodes = filter(n -> !isempty(n.discretization.intervals), continuous_nodes)
#     discretizations_tuples = map(n -> (n, parents(net, n)[3], children(net, n)[3], _discretize(n)), evidence_nodes)
#     for tup in discretizations_tuples
#         node = tup[1]
#         pars = tup[2]
#         chs = tup[3]
#         disc_new = tup[4][1]
#         cont_new = tup[4][2]
#         _remove_node!(net, node)
#         _add_node!(net, disc_new)
#         _add_node!(net, cont_new)
#         add_child!(net, disc_new, cont_new)
#         for par in pars
#             try
#                 add_child!(net, par, disc_new)
#             catch e
#                 @warn "node $(disc_new.name) is a root node and will be added as a child of $(par.name). This is allowed only for network evaluation."
#                 index_par = net.topology[par.name]
#                 index_ch = net.topology[disc_new.name]
#                 net.A[index_par, index_ch] = 1
#             end
#         end
#         for ch in chs
#             add_child!(net, cont_new, ch)
#         end
#         order!(net)
#     end
#     return nothing
# end