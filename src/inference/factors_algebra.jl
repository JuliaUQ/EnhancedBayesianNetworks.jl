# Fix `node` to `state`: slice that dimension out of the table and drop the variable from the factor.
function _restrict(f::Factor, node::Int, state::Int)
    pos = _varpos(f, node)
    if isnothing(pos)
        return f
    end
    newtable = selectdim(f.table, pos, state)
    newvars = deleteat!(copy(f.vars), pos)
    Factor(newvars, newtable)
end

# Marginalise `node` out: sum the table over its dimension and drop the variable from the factor.
function sumout(f::Factor, node::Int)
    pos = _varpos(f, node)
    if isnothing(pos)
        return f
    end
    newtable = sum(f.table, dims=pos)
    newtable = dropdims(newtable; dims=pos)
    newvars = deleteat!(copy(f.vars), pos)
    Factor(newvars, newtable)
end

# Factor product: broadcast-multiply two factors over the union of their variables (each expanded to the
# shared variable layout), yielding a factor over all of them.
function multiply(f1::Factor, f2::Factor)
    # Compute all variables
    allvars = union(f1.vars, f2.vars)
    # Determine variable positon in a Dict to avoid multupli lookups
    allpos = Dict(v => i for (i, v) in enumerate(allvars))
    # Expand and multiply factors
    A = _expand(f1, allvars, allpos)
    B = _expand(f2, allvars, allpos)
    table = A .* B
    return Factor(allvars, table)
end

# Product of many factors, left to right; errors on an empty set (no identity factor is assumed).
function multiply(factors::Vector{<:Factor})
    if isempty(factors)
        error("Cannot multiply an empty factor set")
    end
    return reduce(multiply, factors)
end

# Rescale a factor so its entries sum to 1 (turns an unnormalised marginal into a distribution).
function normalize(f::Factor)
    invZ = inv(sum(f.table))
    Factor(
        f.vars,
        map(x -> x * invZ, f.table)
    )
end

# Reshape a factor's table into the shared `allvars` layout — its dimensions placed at their target
# positions, singleton dimensions elsewhere — so two factors can be broadcast together.
function _expand(f::Factor, allvars::Vector{Int}, allpos::Dict{Int,Int})
    # Factor positions
    positions = [allpos[v] for v in f.vars]
    # Reorder factor dimensions according to allvars
    perm = sortperm(positions)
    A = PermutedDimsArray(f.table, perm)
    # Determine expanded shape
    shape = ones(Int, length(allvars))
    # Reshape the expanded factor
    for (dim, pos) in enumerate(positions[perm])
        shape[pos] = size(A, dim)
    end
    return reshape(A, shape...)
end

# Permute a factor's dimensions so its variables appear in the given order (used to align the result).
function _reorder(f::Factor, vars::Vector{Int})
    isempty(vars) && return f
    isempty(f.vars) && return f
    @assert length(vars) == length(f.vars)
    @assert Set(vars) == Set(f.vars)
    perm = [_varpos(f, v) for v in vars]
    return Factor(vars, permutedims(f.table, perm))
end