# Variable elimination: restrict evidence, then eliminate every non-query, non-evidence variable in the
# given order, _multiply what remains, reorder to the query, and normalise into a posterior.
function _ve(
    factors::Vector{<:Factor},
    order::Vector{Int},
    query_vars::Vector{Int},
    evidence_idx::Vector{Tuple{Int,Int}}
)

    factors = copy(factors)
    apply_evidence!(factors, evidence_idx)
    protected = Set(query_vars)
    for (node, _) ∈ evidence_idx
        push!(protected, node)
    end
    order = [v for v ∈ order if !(v ∈ protected)]
    for var ∈ order
        factors = eliminate_var(factors, var)
    end
    result_query_vars = setdiff(query_vars, first.(evidence_idx))
    result = _multiply(factors)
    result = reorder(result, result_query_vars)
    return normalize(result)
end

# Apply evidence in place: restrict every factor that mentions an observed variable to its observed state.
function apply_evidence!(factors::Vector{<:Factor}, evidence_idx::Vector{Tuple{Int,Int}})
    for (node, state) ∈ evidence_idx
        for i ∈ eachindex(factors)
            if !containsvar(factors[i], node)
                continue
            end
            factors[i] = _restrict(factors[i], node, state)
        end
    end
end

# Eliminate one variable: multiply together the factors that mention it, sum it out, and return the new
# factor set (unmentioned factors untouched).
function eliminate_var(factors::Vector{<:Factor}, var::Int)
    involved = similar(factors, 0)
    remaining = similar(factors, 0)
    for f ∈ factors
        if containsvar(f, var)
            push!(involved, f)
        else
            push!(remaining, f)
        end
    end
    if isempty(involved)
        return factors
    end
    push!(remaining, _sumout(_multiply(involved), var))
    return remaining
end