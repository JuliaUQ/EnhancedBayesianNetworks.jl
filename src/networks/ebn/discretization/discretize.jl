# Split a continuous node into a discrete surrogate (probability mass per interval) and a residual continuous node (the distribution restricted to / approximated on each interval). A root uses its exact distribution and truncation; a child is handled per parent scenario, approximating each interval.
function _discretize(node::ContinuousNode)
    intervals = _format_interval(node)
    name_discrete = Symbol(string(node.name) * "_d")
    discretized_node = DiscreteNode(name_discrete, parents(node))
    new_continuous = ContinuousNode(node.name, [name_discrete])
    if isroot(node)
        map(i -> discretized_node[(name_discrete .=> Symbol(i))] = _discretize(node[], i, node.discretization), intervals)
        map(i -> new_continuous[(name_discrete .=> Symbol(i))] = _truncate(node[], Tuple(i)), intervals)
    else
        for i in intervals
            map((sc) -> discretized_node[(vcat(first.(sc), name_discrete) .=> vcat(last.(sc), Symbol(i)))...] = _discretize(node[(sc)...], i, node.discretization), scenarios(node))
        end
        map(i -> new_continuous[(name_discrete .=> Symbol(i))] = _approximate(i, node.discretization.λ), intervals)
    end
    return (discretized_node, new_continuous)
end

# Turn the node's discretization edges into consecutive [lo, hi] interval pairs, clamped to the distribution's actual support (warning, and trimming/extending, when an edge falls outside it).
function _format_interval(node::ContinuousNode)
    intervals = Float64.(node.discretization.intervals)
    min = node.discretization.intervals[1]
    max = node.discretization.intervals[end]
    lower_bound, upper_bound = _distribution_bounds(node)
    if min > lower_bound
        @warn "node $(repr(node.name)) has minimum intervals value $min > support lower bound $lower_bound. Lower bound will be used as intervals start"
        insert!(intervals, 1, lower_bound)
    end
    if min < lower_bound
        @warn "node $(repr(node.name)) has minimum intervals value $min < support lower bound $lower_bound. Lower bound will be used as intervals start"
        deleteat!(intervals, intervals .<= lower_bound)
        insert!(intervals, 1, lower_bound)
    end
    if max < upper_bound
        @warn "node $(repr(node.name)) has maximum intervals value $max < support upper bound $upper_bound. Upper bound will be used as intervals end"
        push!(intervals, upper_bound)
    end
    if max > upper_bound
        @warn "node $(repr(node.name)) has maximum intervals value $max > support upper bound $upper_bound. Upper bound will be used as intervals end"
        deleteat!(intervals, intervals .>= upper_bound)
        push!(intervals, upper_bound)
    end
    return [[intervals[i], intervals[i + 1]] for i in 1:(length(intervals) - 1)]
end

# Reconstruct a continuous distribution over one interval: Uniform for a bounded interval, a shifted Exponential (rate λ) for a one-sided infinite tail (left tail below an upper edge, right tail above a lower edge).
function _approximate(i::AbstractVector{<:Real}, λ::Real)
    if all(isfinite.(i))
        return Uniform(i...)
    elseif isfinite(last(i))
        return -Exponential(λ) + last(i)
    elseif isfinite(first(i))
        return Exponential(λ) + first(i)
    end
end

# Probability mass assigned to a discretization interval: a CDF difference for a distribution, an interval-valued mass for a ProbabilityBox, and the whole [0, 1] for an (already imprecise) Interval entry.
# The node's discretization strategy comes along because the ProbabilityBox method needs its `technique` and `mesh`; the two methods below ignore it and accept it only so that the node-level `_discretize` can forward it without knowing the entry's type.
function _discretize(
        dist::UnivariateDistribution,
        interval::Vector{<:Real},
        _::AbstractDiscretization
    )
    return cdf(dist, getindex(interval, 2)) - cdf(dist, getindex(interval, 1))
end

function _discretize(_::Interval, _::Vector{<:Real}, _::AbstractDiscretization)
    return Interval(0, 1)
end

function _discretize(dist::ProbabilityBox, interval::Vector{<:Real}, d::AbstractDiscretization)
    return if d.technique === :optimised
        _pbox_mass_optimised(dist, interval; mesh = d.mesh)
    else
        _pbox_mass_free(dist, interval)
    end
end

# Selected by `technique = :distribution_free`: the fallback when the optimisation below proves too slow on some family, and the method that will apply once distribution-free p-boxes land in UncertaintyQuantification.
# Distribution-free p-boxes discretization: valid for *any* CDF lying between the envelopes, so for a parametric p-box it is a conservative approach. The clamp matters, because the lower expression goes negative as soon as the envelopes are far apart and `Interval` only rejects lb > ub, letting a negative probability through.
function _pbox_mass_free(
        dist::ProbabilityBox,
        interval::Vector{<:Real}
    )
    right_p_interval = cdf(dist, getindex(interval, 2))
    left_p_interval = cdf(dist, getindex(interval, 1))
    p_lb = max(0.0, right_p_interval.lb - left_p_interval.ub)
    p_ub = min(1.0, right_p_interval.ub - left_p_interval.lb)
    return Interval(p_lb, p_ub)
end

# Parametric p-box discretization: the interval mass is the range of the mass over the parameter box, evaluated one member of the family at a time. Differencing the CDF envelopes instead would mix members, and evaluating only the corners of the parameter box would miss an interior optimum.
function _pbox_mass_optimised(
        dist::ProbabilityBox,
        interval::Vector{<:Real};
        mesh::Real
    )
    lb, ub = map(v -> Float64.(collect(v)), bounds(dist))
    # internal function for mass over the interval evaluation
    mass = function (θ)
        d = map_to_precise(θ, dist)
        return cdf(d, getindex(interval, 2)) - cdf(d, getindex(interval, 1))
    end
    # A parameter given as a degenerate interval has nothing to optimise over, and the optimiser cannot be handed a zero-width bound, so only the free coordinates are searched and the fixed ones are spliced back in.
    free = findall(i -> lb[i] < ub[i], eachindex(lb))
    # Rebuild a full parameter vector from a searched point, clamped into the box.
    expand = function (y)
        θ = copy(lb)
        θ[free] .= clamp.(y, lb[free], ub[free])
        return θ
    end
    if isempty(free)
        m = mass(lb)
        return Interval(m, m)
    end
    x0 = (lb[free] .+ ub[free]) ./ 2
    opt = OrthoMADS(length(free))
    kwargs = (lowerbound = lb[free], upperbound = ub[free], min_mesh_size = mesh)
    lo = minimize(opt, y -> mass(expand(y)), x0; kwargs...)
    hi = minimize(opt, y -> -mass(expand(y)), x0; kwargs...)
    return Interval(max(0.0, lo.f), min(1.0, -hi.f))
end
