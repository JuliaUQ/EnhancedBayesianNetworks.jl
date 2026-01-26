@auto_hash_equals mutable struct BayesianNetwork2be
    nodes::AbstractVector{Symbol}
    topology::Dict
    A::SparseMatrixCSC

    function BayesianNetwork2be(nodes::AbstractVector{Symbol}, topology::Dict, A::SparseMatrixCSC)
        nodes_names = nodes
        if !allunique(nodes_names)
            error("network nodes names must be unique")
        end
        new(nodes, topology, A)
    end
end

function BayesianNetwork2be(nodes::AbstractVector{Symbol})
    n = length(nodes)
    topology = Dict()
    for (i, n) in enumerate(nodes)
        topology[n] = i
    end
    A = sparse(zeros(n, n))
    return BayesianNetwork2be(nodes, topology, A)
end

BayesianNetwork(nodes::AbstractVector{Symbol}) = BayesianNetwork2be(nodes)

function add_child!(net::BayesianNetwork2be, par::Symbol, ch::Symbol)
    index_par = net.topology[par]
    index_ch = net.topology[ch]
    nodes = net.nodes
    par_node = first(filter(n -> n == par, nodes))
    ch_node = first(filter(n -> n == ch, nodes))
    _verify_no_recursion(par_node, ch_node)
    net.A[index_par, index_ch] = 1
    return nothing
end

function add_child!(net::BayesianNetwork2be, par_index::Int64, ch_index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology)
    par = reverse_dict[par_index]
    ch = reverse_dict[ch_index]
    add_child!(net, par, ch)
end

function order!(net::BayesianNetwork2be)
    if _is_cyclic_dfs(net.A)
        error("network is cyclic!")
    end
    n = net.A.n
    reverse_dict = Dict(value => key for (key, value) in net.topology)
    all_nodes = range(1, n)
    root_indices = findall(map(col -> all(col .== 0), eachcol(net.A)))
    root_nodes = AbstractVector{Symbol}(map(x -> first(filter(j -> j == reverse_dict[x], net.nodes)), root_indices))
    to_be_classified = setdiff(all_nodes, root_indices)
    while !isempty(to_be_classified)
        par_list = map(r -> net.A[:, r].nzind, to_be_classified)
        new_root_indices = findall(map(p -> all(p .∈ [root_indices]), par_list))
        new_root = map(i -> to_be_classified[i], new_root_indices)
        append!(root_indices, new_root)
        new_root_nodes = map(x -> first(filter(j -> j == reverse_dict[x], net.nodes)), new_root)
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
    return nothing
end

function parents(net::BayesianNetwork2be, index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology)
    indices = net.A[:, index].nzind
    names = map(x -> reverse_dict[x], indices)
    nodes = filter(x -> x ∈ names, net.nodes)
    return indices, nodes
end

function parents(net::BayesianNetwork2be, name::Symbol)
    index = net.topology[name]
    parents(net, index)
end

function children(net::BayesianNetwork2be, index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology)
    indices = net.A[index, :].nzind
    names = map(x -> reverse_dict[x], indices)
    nodes = filter(x -> x ∈ names, net.nodes)
    return indices, nodes
end

function children(net::BayesianNetwork2be, name::Symbol)
    index = net.topology[name]
    children(net, index)
end