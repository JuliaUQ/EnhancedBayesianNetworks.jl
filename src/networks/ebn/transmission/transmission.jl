function transfer_continuous_functional_node!(net::EnhancedBayesianNetwork, node::ContinuousFunctionalNode)
    node_children = filter(n -> n.name ∈ children(net, node), net.nodes)
    if isempty(node.discretization) && !isempty(node_children)
        node_parents = filter(n -> n.name ∈ parents(net, node), net.nodes)
        map(ch -> prepend!(ch.models, node.models), node_children)
        remove_node!(net, node)
        add_child!(net, node_parents, node_children)
        return order!(net)
    end
end