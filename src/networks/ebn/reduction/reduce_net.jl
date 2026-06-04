function reduce(net::EnhancedBayesianNetwork, collect::Bool=true)
    order!(net)
    discretize!(net)
    continuous_functional_node = filter(x -> isa(x, ContinuousFunctionalNode), net.nodes)
    map(n -> transfer_continuous_functional_node!(net, n), filter(x -> isempty(x.discretization), continuous_functional_node))
    map(n -> verify_functional_parents(net, n), filter(x -> isa(x, FunctionalNode), net.nodes))
    map(n -> build_simulations!(net, n), filter(x -> isa(x, FunctionalNode), net.nodes))
    map(n -> verify_ancestors(net, n), filter(x -> isa(x, FunctionalNode), net.nodes))
    map(n -> verify_scenarios(net, n), filter(x -> isa(x, FunctionalNode), net.nodes))

    functional_nodes = filter(n -> isa(n, FunctionalNode), net.nodes)
    while !isempty(functional_nodes)
        mapping = map(n -> has_functional_parents(net, n), functional_nodes)
        node2eval = first(functional_nodes[.!mapping])
        evaluated = evaluate(net, node2eval, collect)
        par = parents(net, node2eval)
        chs = children(net, node2eval)
        for n in filter(n -> isa(n, ContinuousNode), filter(n -> n.name ∈ par, net.nodes))
            n_children_functional = filter(n -> isa(n, FunctionalNode), filter(x -> x.name ∈ setdiff(children(net, n), [node2eval.name]), net.nodes))
            if isempty(n_children_functional)
                eliminate_node!(net, n)
            else
                ## removing just the link between node2eval and its continuous parents
                net.A[net.topology[n.name], net.topology[node2eval.name]] = false
                dropzeros!(net.A)
            end
        end
        par = parents(net, node2eval)
        chs = children(net, node2eval)
        remove_node!(net, node2eval)
        add_node!(net, evaluated)
        add_child!(net, par, evaluated.name)
        add_child!(net, evaluated.name, chs)
        functional_nodes = filter(n -> isa(n, FunctionalNode), net.nodes)
    end
    if size(net.A) != (1, 1)
        order!(net)
    end
    return dispatch(net)
end

function has_functional_parents(net::EnhancedBayesianNetwork, node::FunctionalNode)
    any(isa.(filter(n -> n.name ∈ parents(net, node), net.nodes), FunctionalNode))
end

function eliminate_node!(net::EnhancedBayesianNetwork, node::ContinuousNode)
    par = parents(net, node)
    chs = children(net, node)
    remove_node!(net, node)
    add_child!(net, par, chs)
    if iscyclic(net)
        error("Error during node elimination: elimination of node $(repr(node.name)) leads to a cyclic network")
    end
end