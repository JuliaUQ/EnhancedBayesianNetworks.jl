# A discrete factor used by variable elimination: a multidimensional `table` whose k-th dimension is
# indexed by the states of variable `vars[k]` (variables are integer ids from a NetworkSchema, not
# names). The scalar constructor stores `x` as a 0-dim table — a factor over no variables.
struct Factor{T,A<:AbstractArray{T}}
    vars::Vector{Int}
    table::A
end

Factor(vars::Vector{Int}, x::Number) = Factor(vars, fill(x))


# Turn a network (or a single node) into the Factors consumed by variable elimination: each node's CPT
# becomes one factor over the node and its parents, with names mapped to integer ids through the
# NetworkSchema. The BayesianNetwork method returns one factor per node.
function _factorize(node::DiscreteNode, ns::NetworkSchema)
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

function _factorize(bn::BayesianNetwork)
    ns = NetworkSchema(bn)
    map(n -> _factorize(n, ns), bn.nodes)
end

# Position (dimension index) of variable var in factor f, or nothing if f doesn't range over it.
@inline function _varpos(f::Factor, var::Int)
    findfirst(==(var), f.vars)
end

# Whether factor f ranges over variable var.
@inline _containsvar(f::Factor, var::Int) = var ∈ f.vars