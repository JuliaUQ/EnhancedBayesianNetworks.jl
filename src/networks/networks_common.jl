function parents(net::AbstractNetwork, index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology)
    indices = net.A[:, index].nzind
    names = map(x -> reverse_dict[x], indices)
    nodes = filter(x -> x.name ∈ names, net.nodes)
    return indices, names, nodes
end

function parents(net::AbstractNetwork, name::Symbol)
    index = net.topology[name]
    parents(net, index)
end

function parents(net::AbstractNetwork, node::AbstractNode)
    index = net.topology[node.name]
    parents(net, index)
end

function children(net::AbstractNetwork, index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology)
    indices = net.A[index, :].nzind
    names = map(x -> reverse_dict[x], indices)
    nodes = filter(x -> x.name ∈ names, net.nodes)
    return indices, names, nodes
end

function children(net::AbstractNetwork, name::Symbol)
    index = net.topology[name]
    children(net, index)
end

function children(net::AbstractNetwork, node::AbstractNode)
    index = net.topology[node.name]
    children(net, index)
end

function discrete_ancestors(net::AbstractNetwork, node::AbstractNode)
    discrete_parents = filter(x -> isa(x, AbstractDiscreteNode), parents(net, node)[3])
    continuous_parents = filter(x -> isa(x, AbstractContinuousNode), parents(net, node)[3])
    if isempty(continuous_parents)
        return discrete_parents
    end
    return unique([discrete_parents..., mapreduce(x -> discrete_ancestors(net, x), vcat, continuous_parents)
    ...])
end

function _theoretical_scenarios(net::AbstractNetwork, node::AbstractNode)
    par = discrete_ancestors(net, node)
    discrete_parents = filter(x -> isa(x, AbstractDiscreteNode), par)
    function f(par)
        return map(st -> (par.name => st), states(par))
    end
    discrete_parents_combination = Iterators.product(f.(discrete_parents)...)
    discrete_parents_combination = map(t -> [t...], discrete_parents_combination)
    return Dict.(vec(discrete_parents_combination))
end

function add_child!(net::AbstractNetwork, par::Symbol, ch::Symbol)
    index_par = net.topology[par]
    index_ch = net.topology[ch]
    nodes = net.nodes
    par_node = first(filter(n -> n.name == par, nodes))
    ch_node = first(filter(n -> n.name == ch, nodes))
    _verify_no_recursion(par_node, ch_node)
    _verify_root(par_node, ch_node)
    _verify_child(par_node, ch_node)
    _verify_functional_node(par_node, ch_node)
    net.A[index_par, index_ch] = 1
    return nothing
end

function add_child!(net::AbstractNetwork, par_index::Int64, ch_index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology)
    par = reverse_dict[par_index]
    ch = reverse_dict[ch_index]
    add_child!(net, par, ch)
end

function add_child!(net::AbstractNetwork, par_node::AbstractNode, ch_node::AbstractNode)
    par = par_node.name
    ch = ch_node.name
    add_child!(net, par, ch)
end

function order!(net::AbstractNetwork)
    if _is_cyclic_dfs(net.A)
        error("network is cyclic!")
    end
    n = net.A.n
    reverse_dict = Dict(value => key for (key, value) in net.topology)
    all_nodes = range(1, n)
    root_indices = findall(map(col -> all(col .== 0), eachcol(net.A)))
    root_nodes = AbstractVector{AbstractNode}(map(x -> first(filter(j -> j.name == reverse_dict[x], net.nodes)), root_indices))
    to_be_classified = setdiff(all_nodes, root_indices)
    while !isempty(to_be_classified)
        par_list = map(r -> net.A[:, r].nzind, to_be_classified)
        new_root_indices = findall(map(p -> all(p .∈ [root_indices]), par_list))
        new_root = map(i -> to_be_classified[i], new_root_indices)
        append!(root_indices, new_root)
        new_root_nodes = map(x -> first(filter(j -> j.name == reverse_dict[x], net.nodes)), new_root)
        append!(root_nodes, new_root_nodes)
        to_be_classified = setdiff(to_be_classified, new_root)
    end

    ordered_topology = Dict(map(i -> (reverse_dict[i[2]], i[1]), enumerate(root_indices)))
    conversion = Dict(map(i -> (i[2], i[1]), enumerate(root_indices)))
    ordered_matrix = sparse(zeros(n, n))
    for i in range(1, n)
        for j in range(1, n)
            if net.A[i, j] == 1
                ordered_matrix[conversion[i], conversion[j]] = 1
            end
        end
    end

    net.A = ordered_matrix
    net.topology = ordered_topology
    net.nodes = root_nodes
    _verify_net(net)
    return nothing
end

function _remove_node!(net::AbstractNetwork, index::Int64)
    A = net.A[1:end.!=index, 1:end.!=index]
    nodes = deleteat!(net.nodes, index)
    topology_vec = collect(net.topology)
    function f(kv, i)
        if kv[2] > i
            return Pair(kv[1], kv[2] - 1)
        elseif kv[2] != i
            return kv
        end
    end
    topology_vec = map(t -> f(t, index), topology_vec)
    filter!(x -> !isnothing(x), topology_vec)
    topology = Dict(topology_vec)
    net.A = A
    net.topology = topology
    net.nodes = nodes
    return nothing
end

function _remove_node!(net::AbstractNetwork, name::Symbol)
    index = net.topology[name]
    _remove_node!(net, index)
end

function _remove_node!(net::AbstractNetwork, node::AbstractNode)
    index = net.topology[node.name]
    _remove_node!(net, index)
end

function _add_node!(net::AbstractNetwork, node::AbstractNode)
    push!(net.nodes, node)
    net.topology[node.name] = length(net.nodes)
    net.A = hcat(net.A, zeros(net.A.m))
    net.A = vcat(net.A, zeros(net.A.n)')
    return nothing
end

function _verify_net(net::AbstractNetwork)
    map(n -> _verify_child_node(net, n), filter(x -> !isa(x, FunctionalNode), net.nodes))
    functional_nodes = filter(x -> isa(x, FunctionalNode), net.nodes)
    map(fn -> _verify_functional_node(net, fn), functional_nodes)
    return nothing
end

function _verify_functional_node(net::AbstractNetwork, node::FunctionalNode)
    pars = parents(net, node)[3]
    if isempty(pars)
        error("functional node '$(node.name)' must have at least one parent")
    end
    cont_pars = filter(x -> isa(x, AbstractContinuousNode), pars)
    if isempty(cont_pars)
        @warn "functional node '$(node.name)' have no continuous parents. All the simulations will return the same output"
    end
    disc_pars = filter(x -> isa(x, DiscreteNode), pars)
    map(dp -> _non_empty_parameters_vector(net, dp), disc_pars)
    return nothing
end

function _verify_child_node(net::AbstractNetwork, node::DiscreteNode)
    if !isroot(node)
        th_parents_names = Symbol.(names(node.cpt.data[!, Not(node.name, :Π)]))
        if !issetequal(th_parents_names, parents(net, node)[2])
            error("node '$(node.name)''s cpt requires exctly the nodes '$th_parents_names' to be its parents, but provided parents are '$(parents(net, node)[2])'")
        end
        th_scenarios = _theoretical_scenarios(net, node)
        cpt_scenarios = scenarios(node)
        if !issetequal(cpt_scenarios, th_scenarios)
            error("node '$(node.name)' has defined cpt scenarios $(node.cpt) not coherent with the theoretical one $th_scenarios")
        end
    end
    return nothing
end

function _verify_child_node(net::AbstractNetwork, node::ContinuousNode)
    if !isroot(node)
        th_parents_names = Symbol.(names(node.cpt.data[!, Not(:Π)]))
        if !issetequal(th_parents_names, parents(net, node)[2])
            error("node '$(node.name)''s cpt requires exctly the nodes '$th_parents_names' to be its parents, but provided parents are '$(parents(net, node)[2])'")
        end
        th_scenarios = _theoretical_scenarios(net, node)
        cpt_scenarios = scenarios(node)
        if !issetequal(cpt_scenarios, th_scenarios)
            error("node '$(node.name)' has defined cpt scenarios $(node.cpt) not coherent with the theoretical one $th_scenarios")
        end
    end
    return nothing
end

function _non_empty_parameters_vector(net::AbstractNetwork, node::DiscreteNode)
    chs = children(net, node)[3]
    if any(isa.(chs, FunctionalNode)) && isempty(node.parameters)
        error("node '$(node.name)' is a discrete parent of a functional node and cannot have an empty parameters vector")
    end
    return nothing
end