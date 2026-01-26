function _discretize!(net::EnhancedBayesianNetwork)
    continuous_nodes = filter(x -> isa(x, ContinuousNode), net.nodes)
    evidence_nodes = filter(n -> !isempty(n.discretization.intervals), continuous_nodes)
    discretizations_tuples = map(n -> (n, parents(net, n)[3], children(net, n)[3], _discretize(n)), evidence_nodes)
    for tup in discretizations_tuples
        node = tup[1]
        pars = tup[2]
        chs = tup[3]
        disc_new = tup[4][1]
        cont_new = tup[4][2]
        _remove_node!(net, node)
        _add_node!(net, disc_new)
        _add_node!(net, cont_new)
        add_child!(net, disc_new, cont_new)
        for par in pars
            try
                add_child!(net, par, disc_new)
            catch e
                @warn "node $(disc_new.name) is a root node and will be added as a child of $(par.name). This is allowed only for network evaluation."
                index_par = net.topology[par.name]
                index_ch = net.topology[disc_new.name]
                net.A[index_par, index_ch] = 1
            end
        end
        for ch in chs
            add_child!(net, cont_new, ch)
        end
        order!(net)
    end
    return nothing
end

function _discretize(node::ContinuousNode)
    intervals = _format_interval(node)
    states_symbols = Symbol.(intervals)
    n = length(states_symbols)
    if isempty(scenarios(node))
        m = 1
    else
        m = length(scenarios(node))
    end
    name_discrete = Symbol(string(node.name) * "_d")
    new_cpt_disc = repeat(node.cpt.data[!, Not(:Π)], inner=n)
    new_cpt_disc[!, name_discrete] = repeat(states_symbols, m)
    probs = map(dist -> _discretize(dist, intervals), node.cpt.data[!, :Π])
    new_cpt_disc[!, :Π] = collect(Iterators.flatten(probs))
    if typeof(node.cpt).parameters[1] == UnivariateDistribution
        cpt_disc = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(new_cpt_disc)
    else
        cpt_disc = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}(new_cpt_disc)
    end
    discrete_node = DiscreteNode(name_discrete, cpt_disc)
    if isroot(node)
        distribution = mapreduce(dist -> _truncate.(Ref(dist), intervals), vcat, node.cpt.data[!, :Π])
        new_cpt_cont = DataFrame(name_discrete => states_symbols, :Π => distribution)
        cpt_cont = ContinuousConditionalProbabilityTable{typeof(node.cpt).parameters[1]}(new_cpt_cont)
        continuous_node = ContinuousNode(node.name, cpt_cont)
    else
        distribution = _approximate.(intervals, node.discretization.sigma)
        new_cpt_cont = DataFrame(name_discrete => states_symbols, :Π => distribution)
        cpt_cont = ContinuousConditionalProbabilityTable{PreciseContinuousInput}(new_cpt_cont)
        continuous_node = ContinuousNode(node.name, cpt_cont)
    end
    return [discrete_node, continuous_node]
end

## Auxiliary function
function _discretize(dist::UnivariateDistribution, intervals::Vector)
    return cdf.(dist, getindex.(intervals, 2)) .- cdf.(dist, getindex.(intervals, 1))
end

function _discretize(dist::UnamedProbabilityBox, intervals::Vector)
    p_box = ProbabilityBox{first(typeof(dist).parameters)}(dist.parameters, :temp)
    right_bounds = cdf.(Ref(p_box), getindex.(intervals, 2))
    left_bounds = cdf.(Ref(p_box), getindex.(intervals, 1))
    map((r, l) -> (minimum([r.lb - l.lb, r.ub - l.ub]), maximum([r.lb - l.lb, r.ub - l.ub])), right_bounds, left_bounds)
end

function _discretize(_::Tuple{T,T}, intervals::Vector) where {T<:Real}
    repeat([(0, 1)], length(intervals))
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