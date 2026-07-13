"""
    reduce(net::EnhancedBayesianNetwork, collect::Bool=true)

Transform an enhanced Bayesian network into a purely discrete one ready for inference, returning a
[`BayesianNetwork`](@ref) (all nodes precise) or a [`CredalNetwork`](@ref) (some imprecise). The
network is ordered, its continuous nodes discretized, and its functional nodes evaluated in dependency
order — each functional node is simulated over the scenario grid of its discrete ancestors and replaced
by the resulting node, eliminating the continuous parents that fed only it. With `collect=true` the
intermediate simulation samples are kept on the evaluated nodes' `results`.

# Examples
```julia
W = DiscreteNode(:W, [:sunny => [Parameter(1.0, :W)], :cloudy => [Parameter(2.0, :W)]])
W[:W => :sunny] = 0.5; W[:W => :cloudy] = 0.5
X = ContinuousNode(:X, Uniform(-1, 1), ExactDiscretization([-1.0, 0.0, 1.0]))
model = Model(df -> df.X .+ df.W, :Y)
F = DiscreteFunctionalNode(:F, [model], df -> df.Y, MonteCarlo(200))

ebn = EnhancedBayesianNetwork([W, X, F])
add_child!(ebn, :W, :F); add_child!(ebn, :X, :F); order!(ebn)
reduce(ebn)                                 # -> BayesianNetwork
```
"""
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
        # evaluate a functional node whose parents are all non-functional (its inputs are ready)
        mapping = map(n -> has_functional_parents(net, n), functional_nodes)
        node2eval = first(functional_nodes[.!mapping])
        evaluated = evaluate(net, node2eval, collect)
        par = parents(net, node2eval)
        for n in filter(n -> isa(n, ContinuousNode), filter(n -> n.name ∈ par, net.nodes))
            n_children_functional = filter(n -> isa(n, FunctionalNode), filter(x -> x.name ∈ setdiff(children(net, n), [node2eval.name]), net.nodes))
            if isempty(n_children_functional)
                # a continuous parent feeding only this node is no longer needed → drop it
                eliminate_node!(net, n)
            else
                # a continuous parent still feeds other functional nodes → cut only this edge
                net.A[net.topology[n.name], net.topology[node2eval.name]] = false
                dropzeros!(net.A)
            end
        end
        # swap the functional node for its evaluated (discrete/continuous) result, keeping its wiring
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

# True if any parent of this functional node is itself a functional node (so it can't be evaluated yet).
function has_functional_parents(net::EnhancedBayesianNetwork, node::FunctionalNode)
    any(isa.(filter(n -> n.name ∈ parents(net, node), net.nodes), FunctionalNode))
end

# Remove a continuous node and reconnect its parents directly to its children; errors if that creates a cycle.
function eliminate_node!(net::EnhancedBayesianNetwork, node::ContinuousNode)
    par = parents(net, node)
    chs = children(net, node)
    remove_node!(net, node)
    add_child!(net, par, chs)
    if iscyclic(net)
        error("Error during node elimination: elimination of node $(repr(node.name)) leads to a cyclic network")
    end
end