function apply_evidence!(factors::Vector{<:Factor}, evidence_idx::Vector{Tuple{Int,Int}})
    for (node, state) in evidence_idx
        for i in eachindex(factors)
            if !containsvar(factors[i], node)
                continue
            end
            factors[i] = restrict(factors[i], node, state)
        end
    end
end

function eliminate_var(factors::Vector{<:Factor}, var::Int)
    involved = similar(factors, 0)
    remaining = similar(factors, 0)
    for f in factors
        if containsvar(f, var)
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