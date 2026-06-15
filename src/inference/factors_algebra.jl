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
    # Reshape facots in the new allvars-space

end