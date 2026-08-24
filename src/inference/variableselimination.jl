# Variable elimination: restrict evidence, then eliminate every non-query, non-evidence variable in the
# given order, multiply what remains, reorder to the query, and normalise into a posterior.
# `progress` (default false) draws a bar over the eliminated variables; `infer` sets it from `isinteractive()`.
function _ve(
        factors::Vector{<:Factor},
        order::Vector{Int},
        query_vars::Vector{Int},
        evidence_idx::Vector{Tuple{Int, Int}};
        progress::Bool = false
    )

    factors = copy(factors)
    _apply_evidence!(factors, evidence_idx)
    protected = Set(query_vars)
    for (node, _) in evidence_idx
        push!(protected, node)
    end
    order = [v for v in order if !(v ∈ protected)]
    p = Progress(length(order); desc = "Eliminating variables ", enabled = progress)
    for var in order
        factors = _eliminate_var(factors, var)
        next!(p)
    end
    result_query_vars = setdiff(query_vars, first.(evidence_idx))
    result = multiply(factors)
    result = _reorder(result, result_query_vars)
    return normalize(result)
end

# Apply evidence in place: restrict every factor that mentions an observed variable to its observed state.
function _apply_evidence!(factors::Vector{<:Factor}, evidence_idx::Vector{Tuple{Int, Int}})
    for (node, state) in evidence_idx
        for i in eachindex(factors)
            if !_containsvar(factors[i], node)
                continue
            end
            factors[i] = _restrict(factors[i], node, state)
        end
    end
    return
end

# Eliminate one variable: multiply together the factors that mention it, sum it out, and return the new
# factor set (unmentioned factors untouched).
function _eliminate_var(factors::Vector{<:Factor}, var::Int)
    involved = similar(factors, 0)
    remaining = similar(factors, 0)
    for f in factors
        if _containsvar(f, var)
            push!(involved, f)
        else
            push!(remaining, f)
        end
    end
    if isempty(involved)
        return factors
    end
    push!(remaining, sumout(multiply(involved), var))
    return remaining
end
