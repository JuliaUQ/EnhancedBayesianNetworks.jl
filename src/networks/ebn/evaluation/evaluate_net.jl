function evaluate!(net::EnhancedBayesianNetwork, check::Bool=true, collect_samples::Bool=true)
    discretize!(net)
    _transfer_continuous!(net)
    functional_nodes = filter(x -> isa(x, FunctionalNode), net.nodes)
    while !isempty(functional_nodes)
        first_node = first(functional_nodes)
        continuous_nodes_2_eliminate = filter(x -> isa(x, ContinuousNode), parents(net, first_node)[3])
        eliminable_node = map(x -> _is_eliminable(net, x), continuous_nodes_2_eliminate)
        if any(.!eliminable_node) && check
            names = [i.name for i in continuous_nodes_2_eliminate[.!eliminable_node]]
            error("nodes elimnation algorithm will lead to a cyclic network when elimnating node/s $names")
        end
        local evaluated_node
        try
            evaluated_node = _evaluate_node(net, first_node, collect_samples)
        catch e
            if isa(e, AssertionError)
                error("node $(first_node.name) has as simulation $(first_node.simulation), but its imprecise parents will be discretized and approximated with Uniform and Exponential assumption, therefore are no longer imprecise. A prices simulation technique must be selected!")
            else
                pars = parents(net, first_node)[3]
                imprecise_parents = pars[map(!, isprecise.(pars))]
                names = [i.name for i in imprecise_parents]
                error("node $(getproperty(first_node, :name)) has $(getproperty(first_node, :simulation)) as simulation technique, but have $names as imprecise parent/s. DoubleLoop or RandomSlicing technique must be employeed instead")
            end
        end
        index = findfirst(==(first_node), net.nodes)
        net.nodes[index] = evaluated_node
        discretize!(net)
        _transfer_continuous!(net)
        functional_nodes = filter(x -> isa(x, FunctionalNode), net.nodes)
    end
end

function evaluate_with_envelopes(net::EnhancedBayesianNetwork)
    envelopes = markov_envelope(net)
    envelopes = map(x -> _add_root2envelope(net, x), envelopes)
    ebns = map(x -> _build_envelope_edges(net, x), envelopes)
    map(ebn -> evaluate!(ebn), ebns)
    return ebns
end

function _is_eliminable(net::EnhancedBayesianNetwork, node::AbstractNode)
    if !isa(node, AbstractContinuousNode)
        error("node elimination algorithm is for continuous nodes and $(node.name) is discrete")
    end
    index = net.topology[node.name]
    test_matrix = deepcopy(net.A)
    pars = parents(net, index)[1]
    map(x -> test_matrix[x, index] = 0, pars)
    map(x -> test_matrix[index, x] = 1, pars)
    chs = children(net, index)[1]
    map(x -> test_matrix[index, x] = 0, chs)
    map(x -> test_matrix[x, index] = 1, chs)
    !_is_cyclic_dfs(test_matrix)
end

function _is_eliminable(net::EnhancedBayesianNetwork, index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology)
    index = findfirst(x -> x.name == reverse_dict[index], net.nodes)
    _is_eliminable(net, net.nodes[index])
end

function _is_eliminable(net::EnhancedBayesianNetwork, name::Symbol)
    index = findfirst(x -> x.name == name, net.nodes)
    _is_eliminable(net, net.nodes[index])
end

function _add_root2envelope(net::EnhancedBayesianNetwork, envelope::AbstractVector{<:AbstractNode})
    parents_vector = map(x -> parents(net, x)[3], envelope)
    is_not_in = map(x -> [i ∉ envelope for i in x], parents_vector)
    while any(collect(Iterators.Flatten(is_not_in)))
        missing_nodes = map((x, y) -> x[y], parents_vector, is_not_in)
        missing_nodes = unique(collect(Iterators.Flatten(missing_nodes)))
        envelope = append!(envelope, missing_nodes)
        parents_vector = map(x -> parents(net, x)[3], envelope)
        is_not_in = map(x -> [i ∉ envelope for i in x], parents_vector)
    end
    return envelope
end

function _build_envelope_edges(net::EnhancedBayesianNetwork, envelope::AbstractVector{<:AbstractNode})
    ebn = EnhancedBayesianNetwork(envelope)
    for node in envelope
        par = parents(net, node)[3]
        for p in par
            if p ∈ envelope
                add_child!(ebn, p, node)
            end
        end
    end
    order!(ebn)
    return ebn
end