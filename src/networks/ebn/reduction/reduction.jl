
function reduce!(net::EnhancedBayesianNetwork, check::Bool=true, collect_samples::Bool=true)
    if !isempty(filter(x -> isa(x, FunctionalNode), net.nodes))
        evaluate!(net, check, collect_samples)
    end
    cont_nodes = filter(x -> isa(x, ContinuousNode), net.nodes)
    map(x -> _eliminate_continuous_node!(net, x), cont_nodes)
    return nothing
end

function _eliminate_continuous_node!(net::EnhancedBayesianNetwork, node::ContinuousNode)
    parents_indices = parents(net, node)[1]
    children_indices = children(net, node)[1]
    pairs = [(p, c) for p in parents_indices, c in children_indices]
    map(i -> net.A[i[1], i[2]] = 1, pairs)
    _remove_node!(net, node)
end