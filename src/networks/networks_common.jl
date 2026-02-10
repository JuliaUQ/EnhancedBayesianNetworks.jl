iscyclic(net::AbstractNetwork) = iscyclic(net.A)

isconnected(net::AbstractNetwork) = isconnected(net.A)

function parents(net::AbstractNetwork, name::Symbol)
    index = net.topology[name]
    parents_index = net.A[:, index].nzind
    reverse_dict = Dict(value => key for (key, value) in net.topology)

    return map(i -> reverse_dict[i], parents_index)
end

parents(net::AbstractNetwork, node::AbstractNode) = parents(net, node.name)

function children(net::AbstractNetwork, name::Symbol)
    index = net.topology[name]
    children_index = net.A[index, :].nzind
    reverse_dict = Dict(value => key for (key, value) in net.topology)

    return map(i -> reverse_dict[i], children_index)
end

children(net::AbstractNetwork, node::AbstractNode) = children(net, node.name)

function verify_parents(net::AbstractNetwork, node::AbstractNode) ## verify if all the parents in the CPT have been added via add_child!
    if isa(node, FunctionalNode)
        return nothing
    else
        cpt_parents = parents(node)
        net_parents = parents(net, node.name)
        only_in_cpt = setdiff(cpt_parents, net_parents)
        if !isempty(only_in_cpt)
            error("Invalid CPT: node $(node.name) has node(s) '$only_in_cpt' defined in the CPT only, but they have not been added via add_child!")
        end
    end
end

function verify_scenarios(net::AbstractNetwork, node::DiscreteNode)
    par = filter(n -> n.name ∈ parents(node), net.nodes)
    v = vcat(par, node)
    theoretical_scenarios = vec(collect(Iterators.product(states.(v)...)))
    filtering_elements = map(th_s -> ([i.name for i in v] .=> th_s), theoretical_scenarios)
    for filtering_element in filtering_elements
        if isempty(filter(node.cpt, filtering_element...))
            error("Invalid CPT: node $(node.name) is missing the following scenario $(filtering_element)")
        end
    end
end

function verify_exhaustiveness(net::AbstractNetwork, node::DiscreteNode)
    par = filter(n -> n.name ∈ parents(node), net.nodes)
    theoretical_scenarios = vec(collect(Iterators.product(states.(par)...)))
    filtering_elements = map(th_s -> ([i.name for i in par] .=> th_s), theoretical_scenarios)
    if isprecise(node)
        for filtering_element in filtering_elements
            cumulative_prob = sum(filter(node.cpt, filtering_element...).Π)
            if cumulative_prob != 1
                if isapprox(cumulative_prob, 1, atol=0.01)
                    @warn "node $(node.name) has CPT values '$(filter(node.cpt, filtering_element...).Π)' for the scenario $filtering_element and will be normalized!"
                    filter(node.cpt, filtering_element...)[!, :Π] ./= cumulative_prob
                else
                    error("Invalid CPT: node $(node.name) has CPT values '$(filter(node.cpt, filtering_element...).Π)' not exhaustive and mutually exclusive for the scenario $filtering_element")
                end
            end
        end
    else
        for filtering_element in filtering_elements
            lb_sum, ub_sum = EnhancedBayesianNetworks.sum_intervals_and_float(filter(node.cpt, filtering_element...).Π...)
            if lb_sum > 1
                error("Invalid CPT:  node $(node.name) has CPT values '$(filter(node.cpt, filtering_element...).Π)' for the scenario $filtering_element, the sum of lower bound values must be less than 1")
            elseif ub_sum < 1
                error("Invalid CPT:  node $(node.name) has CPT values '$(filter(node.cpt, filtering_element...).Π)' for the scenario $filtering_element, the sum of upper bound values must be greater than 1")
            end
        end
    end
end

function verify_functional_parents(net::AbstractNetwork, node::FunctionalNode) ## Discrete Parents must have a non empty parameters attribute
    par = filter(n -> n.name ∈ parents(net, node), net.nodes)
    discrete_par = filter(x -> isa(x, DiscreteNode), par)
    cont_par = filter(x -> isa(x, ContinuousNode), par)

    for dp in discrete_par
        if isempty(dp.parameters)
            error("Invalid network: node $(dp.name) is a parent for the FuctionalNode $(node.name) and cannot have an empty parameters attribute")
        end
    end
    if isempty(cont_par)
        @warn "node $(node.name) is a FunctionalNode with no continuous parents. Resulting failure probabilities are Boolean"
    end
    if isempty(discrete_par)
        @warn "node $(node.name) is a FunctionalNode with no discrete parents. Resulting network is a standard reliability analysis"
    end
end

function markov_blanket(net::AbstractNetwork, node::Symbol)
    blanket = Symbol[]
    for child in children(net, node)
        append!(blanket, setdiff(parents(net, child), [node]))
        push!(blanket, child)
    end
    append!(blanket, parents(net, node))
    return unique!(blanket)
end

markov_blanket(net::AbstractNetwork, node::AbstractNode) = markov_blanket(net, node.name)

# function _remove_node!(net::AbstractNetwork, index::Int64)
#     A = net.A[1:end.!=index, 1:end.!=index]
#     nodes = deleteat!(net.nodes, index)
#     topology_vec = collect(net.topology)
#     function f(kv, i)
#         if kv[2] > i
#             return Pair(kv[1], kv[2] - 1)
#         elseif kv[2] != i
#             return kv
#         end
#     end
#     topology_vec = map(t -> f(t, index), topology_vec)
#     filter!(x -> !isnothing(x), topology_vec)
#     topology = Dict(topology_vec)
#     net.A = A
#     net.topology = topology
#     net.nodes = nodes
#     return nothing
# end

# function _remove_node!(net::AbstractNetwork, name::Symbol)
#     index = net.topology[name]
#     _remove_node!(net, index)
# end

# function _remove_node!(net::AbstractNetwork, node::AbstractNode)
#     index = net.topology[node.name]
#     _remove_node!(net, index)
# end

# function _add_node!(net::AbstractNetwork, node::AbstractNode)
#     push!(net.nodes, node)
#     net.topology[node.name] = length(net.nodes)
#     net.A = hcat(net.A, zeros(net.A.m))
#     net.A = vcat(net.A, zeros(net.A.n)')
#     return nothing
# end