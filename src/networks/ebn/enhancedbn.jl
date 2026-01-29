mutable struct EnhancedBayesianNetwork <: AbstractNetwork
    nodes::AbstractVector{<:AbstractNode}
    topology::Dict
    A::SparseMatrixCSC

    function EnhancedBayesianNetwork(
        nodes::AbstractVector{<:AbstractNode},
        topology::Dict,
        A::SparseMatrixCSC
    )
        node_names = map(i -> i.name, nodes)
        dups = not_unique_elements(node_names)
        if !isempty(dups)
            error("Invalid eBN: duplicate node names $dups")
        end
        discrete_nodes = filter(x -> isa(x, DiscreteNode), nodes)
        if !isempty(discrete_nodes)
            states_list = mapreduce(i -> states(i), vcat, discrete_nodes)
            dups = not_unique_elements(states_list)
            if !isempty(dups)
                error("Invalid eBN: duplicate node states $dups")
            end
        end
        new(nodes, topology, A)
    end
end

function EnhancedBayesianNetwork(nodes::AbstractVector{<:AbstractNode})
    n = length(nodes)
    topology = Dict()
    for (i, n) in enumerate(nodes)
        topology[n.name] = i
    end
    A = sparse(zeros(n, n))
    return EnhancedBayesianNetwork(nodes, topology, A)
end

function add_child!(
    net::AbstractNetwork,
    par::Union{<:AbstractNode,Vector{<:AbstractNode}},
    ch::Union{<:AbstractNode,Vector{<:AbstractNode}}
)
    parents = wrap(par)
    children = wrap(ch)
    ## verify No recursion
    verify_no_recursion(parents, children)
    ## verify Discrete parent nodes
    discrete_par = filter(x -> isa(x, DiscreteNode), parents)
    map(dp -> verify_discrete(dp, children), discrete_par)
    ## verify Continuous and Functional parent nodes
    continuous_par = filter(x -> isa(x, ContinuousNode), parents)
    functional_par = filter(x -> isa(x, FunctionalNode), parents)
    cont_and_fun_par = vcat(continuous_par, functional_par)
    map(cfp -> verify_continuous_and_functional(cfp, children), cont_and_fun_par)

    pidx = getindex.(Ref(net.topology), getfield.(parents, :name))
    cidx = getindex.(Ref(net.topology), getfield.(children, :name))
    net.A[pidx, cidx] .= 1
end

function add_child!(
    net::AbstractNetwork,
    par::Union{Symbol,Vector{Symbol}},
    ch::Union{Symbol,Vector{Symbol}}
)
    parents = wrap(par)
    children = wrap(ch)
    par_nodes = filter(x -> x.name ∈ parents, net.nodes)
    ch_nodes = filter(x -> x.name ∈ children, net.nodes)
    add_child!(net, par_nodes, ch_nodes)
end

function _is_cyclic_dfs(A)
    n = size(A, 1)  # Number of nodes
    visited = fill(false, n)
    recStack = fill(false, n)
    function dfs(v)
        visited[v] = true
        recStack[v] = true
        for neighbor in 1:n
            if A[v, neighbor] != 0  # there's an edge from v to neighbor
                if !visited[neighbor]  # If neighbor hasn't been visited, visit it
                    if dfs(neighbor)
                        return true  # Cycle detected
                    end
                elseif recStack[neighbor]  # If neighbor is in recStack, cycle detected
                    return true
                end
            end
        end
        recStack[v] = false
        return false
    end
    for node in 1:n
        if !visited[node]  # Only visit unvisited nodes
            if dfs(node)  # Cycle detected
                return true
            end
        end
    end
    return false  # No cycle found
end

function markov_blanket(net::EnhancedBayesianNetwork, index::Int64)
    blanket = []
    reverse_dict = Dict(value => key for (key, value) in net.topology)
    for child in children(net, index)[1]
        append!(blanket, parents(net, child)[1])
        push!(blanket, child)
    end
    append!(blanket, parents(net, index)[1])
    indices = unique(setdiff(blanket, [index]))
    names = map(x -> reverse_dict[x], indices)
    nodes = filter(x -> x.name ∈ names, net.nodes)
    return indices, names, nodes
end

function markov_blanket(net::EnhancedBayesianNetwork, name::Symbol)
    index = net.topology[name]
    markov_blanket(net, index)
end

function markov_blanket(net::EnhancedBayesianNetwork, node::AbstractNode)
    index = net.topology[node.name]
    markov_blanket(net, index)
end

function _get_markov_group(net::EnhancedBayesianNetwork, node::AbstractNode)
    fun = (ebn, n) -> unique(vcat(n, mapreduce(x -> filter(x -> isa(x, AbstractContinuousNode), markov_blanket(ebn, node)[3]), vcat, n)))
    list = [node]
    new_list = fun(net, list)
    while !issetequal(list, new_list)
        list = new_list
        new_list = fun(net, new_list)
    end
    return new_list
end

function markov_envelope(net::EnhancedBayesianNetwork)
    cont_nodes = filter(x -> isa(x, AbstractContinuousNode), net.nodes)
    Xm_groups = map(x -> _get_markov_group(net, x), cont_nodes)
    markov_envelopes = unique.(mapreduce.(x -> push!(markov_blanket(net, x)[3], x), vcat, Xm_groups))
    # check when a vector is included into another
    sorted_envelopes = sort(markov_envelopes, by=length, rev=true)
    final_envelopes = []
    while length(sorted_envelopes) >= 1
        if length(sorted_envelopes) == 1
            append!(final_envelopes, sorted_envelopes)
            popfirst!(sorted_envelopes)
        else
            envelope = first(sorted_envelopes)
            to_compare_list = sorted_envelopes[2:end]
            is_excluded = map(to_compare -> any(to_compare .∉ [envelope]), to_compare_list)
            sorted_envelopes = to_compare_list[is_excluded]
            push!(final_envelopes, envelope)
        end
    end
    return final_envelopes
end