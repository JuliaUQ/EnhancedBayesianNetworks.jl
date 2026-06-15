struct NetworkSchema
    node_to_idx::Dict{Symbol,Int}
    idx_to_node::Vector{Symbol}
    state_to_idx::Vector{Dict{Symbol,Int}}
    idx_to_state::Vector{Vector{Symbol}}

    function NetworkSchema(bn::BayesianNetwork)
        idx_to_node = getproperty.(bn.nodes, :name)
        node_to_idx = Dict(
            v => i
            for (i, v) in enumerate(idx_to_node)
        )
        idx_to_state = states.(bn.nodes)
        state_to_idx = [
            Dict(state => i
                 for (i, state) in enumerate(sts))
            for sts in idx_to_state
        ]
        new(node_to_idx, idx_to_node, state_to_idx, idx_to_state)
    end
end

struct Factor{T,A<:AbstractArray{T}}
    vars::Vector{Int}
    table::A
end

function factorize(node::DiscreteNode, ns::NetworkSchema)
    var_names = Symbol.(names(node.cpt.data))[1:end-1]
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

function varpos(f::Factor, var::Int)
    findfirst(==(var), f.vars)
end