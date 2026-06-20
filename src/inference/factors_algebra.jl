function restrict(f::Factor, node::Int, state::Int)
    pos = varpos(f, node)
    if isnothing(pos)
        return f
    end
    newtable = selectdim(f.table, pos, state)
    newvars = deleteat!(copy(f.vars), pos)
    Factor(newvars, newtable)
end

function sumout(f::Factor, node::Int)
    pos = varpos(f, node)
    if isnothing(pos)
        return f
    end
    newtable = sum(f.table, dims=pos)
    newtable = dropdims(newtable; dims=pos)
    newvars = deleteat!(copy(f.vars), pos)
    Factor(newvars, newtable)
end

function multiply(f1::Factor, f2::Factor)
    # Compute all variables
    allvars = union(f1.vars, f2.vars)
    # Determine variable positon in a Dict to avoid multupli lookups
    allpos = Dict(v => i for (i, v) in enumerate(allvars))
    # Expand and multiply factors
    A = expand(f1, allvars, allpos)
    B = expand(f2, allvars, allpos)
    return Factor(
        allvars,
        A .* B
    )
end

function multiply(factors::Vector{<:Factor})
    if isempty(factors)
        error("Cannot multiply an empty factor set")
    end
    return reduce(multiply, factors)
end

function normalize(f::Factor)
    invZ = inv(sum(f.table))
    Factor(
        f.vars,
        map(x -> x * invZ, f.table)
    )
end

function expand(f::Factor, allvars::Vector{Int}, allpos::Dict{Int,Int})
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

function reorder(f::Factor, vars::Vector{Int})
    perm = [varpos(f, v) for v in vars]
    return Factor(vars, permutedims(f.table, perm))
end