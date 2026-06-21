struct Factor{T,A<:AbstractArray{T}}
    vars::Vector{Int}
    table::A
end

Factor(vars::Vector{Int}, x::Number) = Factor(vars, fill(x))

function factorize(node::DiscreteNode, ns::NetworkSchema)
    var_names = Symbol.(names(node.cpt.data))[1:(end-1)]
    var_idxs = map(vn -> ns.node_to_idx[vn], var_names)
    dims = map(v_id -> length(ns.idx_to_state[v_id]), var_idxs)
    table = zeros(Float64, dims...)
    for row in eachrow(node.cpt.data)
        idxs = map(var_names, var_idxs) do name, idx
            ns.state_to_idx[idx][row[name]]
        end
        table[idxs...] = row.Π
    end
    return Factor(var_idxs, table)
end

function factorize(bn::BayesianNetwork)
    ns = NetworkSchema(bn)
    map(n -> factorize(n, ns), bn.nodes)
end

@inline function varpos(f::Factor, var::Int)
    findfirst(==(var), f.vars)
end

@inline containsvar(f::Factor, var::Int) = var ∈ f.vars