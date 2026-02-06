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
    A = spzeros(Bool, n, n)
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
    map(p -> verify_no_recursion(p, children), parents)
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
    net.A[pidx, cidx] .= true
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

function order!(net::AbstractNetwork)
    if iscyclic(net.A)
        error("Invalid eBN: network is cyclic!")
    end

    if !isconnected(net.A)
        error("Invalid eBN: network is not connected")
    end

    all_indices = range(1, net.A.n)
    root_nodes = net.nodes[isroot.(net.nodes)]
    root_indices = map(rn -> net.topology[rn.name], root_nodes)
    to_be_classified = setdiff(all_indices, root_indices)
    while !isempty(to_be_classified)
        par_list_indices = map(r -> net.A[:, r].nzind, to_be_classified)
        new_root_indices = findall(map(p -> all(p .∈ [root_indices]), par_list_indices))
        new_root_indices = map(i -> to_be_classified[i], new_root_indices)
        append!(root_indices, new_root_indices)
        to_be_classified = setdiff(to_be_classified, new_root_indices)
    end
    ordered_indices = root_indices
    reverse_dict = Dict(value => key for (key, value) in net.topology)
    ordered_topology = Dict(map(i -> (reverse_dict[i[2]], i[1]), enumerate(ordered_indices)))
    conversion = Dict(map(i -> (i[2], i[1]), enumerate(root_indices)))
    ordered_matrix = spzeros(Bool, net.A.n, net.A.n)
    for i in range(1, net.A.n)
        for j in range(1, net.A.n)
            if net.A[i, j] == true
                ordered_matrix[conversion[i], conversion[j]] = true
            end
        end
    end
    net.A = ordered_matrix
    net.topology = ordered_topology
    map(n -> verify_parents(net, n), net.nodes)
    map(n -> verify_scenarios(net, n), filter(x -> isa(x, DiscreteNode), net.nodes))
    map(n -> verify_exhaustiveness(net, n), filter(x -> isa(x, DiscreteNode), net.nodes))
    map(n -> verify_functional_parents(net, n), filter(x -> isa(x, FunctionalNode), net.nodes))
    return nothing
end

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

## Verification Steps

function verify_parents(net::AbstractNetwork, node::AbstractNode) ## verify if all the parents in the CPT have been added via add_child!
    if isa(node, FunctionalNode)
        return nothing
    else
        cpt_parents = parents(node)
        net_parents = parents(net, node.name)
        only_in_cpt = setdiff(cpt_parents, net_parents)
        if !isempty(only_in_cpt)
            error("Invalid eBN: node $(node.name) has node(s) '$only_in_cpt' defined in the CPT only, but they have not been added via add_child!")
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
            error("Invalid eBN: node $(node.name) is missing the following scenario $(filtering_element)")
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
            error("Invalid eBN: node $(dp.name) is a parent for the FuctionalNode $(node.name) and cannot have an empty parameters attribute")
        end
    end
    if isempty(cont_par)
        @warn "node $(node.name) is a FunctionalNode with no continuous parents. Resulting failure probabilities are Boolean"
    end
    if isempty(discrete_par)
        @warn "node $(node.name) is a FunctionalNode with no discrete parents. Resulting eBN is a standard reliability analysis"
    end
end









# function markov_blanket(net::EnhancedBayesianNetwork, index::Int64)
#     blanket = []
#     reverse_dict = Dict(value => key for (key, value) in net.topology)
#     for child in children(net, index)[1]
#         append!(blanket, parents(net, child)[1])
#         push!(blanket, child)
#     end
#     append!(blanket, parents(net, index)[1])
#     indices = unique(setdiff(blanket, [index]))
#     names = map(x -> reverse_dict[x], indices)
#     nodes = filter(x -> x.name ∈ names, net.nodes)
#     return indices, names, nodes
# end

# function markov_blanket(net::EnhancedBayesianNetwork, name::Symbol)
#     index = net.topology[name]
#     markov_blanket(net, index)
# end

# function markov_blanket(net::EnhancedBayesianNetwork, node::AbstractNode)
#     index = net.topology[node.name]
#     markov_blanket(net, index)
# end

# function _get_markov_group(net::EnhancedBayesianNetwork, node::AbstractNode)
#     fun = (ebn, n) -> unique(vcat(n, mapreduce(x -> filter(x -> isa(x, AbstractContinuousNode), markov_blanket(ebn, node)[3]), vcat, n)))
#     list = [node]
#     new_list = fun(net, list)
#     while !issetequal(list, new_list)
#         list = new_list
#         new_list = fun(net, new_list)
#     end
#     return new_list
# end

# function markov_envelope(net::EnhancedBayesianNetwork)
#     cont_nodes = filter(x -> isa(x, AbstractContinuousNode), net.nodes)
#     Xm_groups = map(x -> _get_markov_group(net, x), cont_nodes)
#     markov_envelopes = unique.(mapreduce.(x -> push!(markov_blanket(net, x)[3], x), vcat, Xm_groups))
#     # check when a vector is included into another
#     sorted_envelopes = sort(markov_envelopes, by=length, rev=true)
#     final_envelopes = []
#     while length(sorted_envelopes) >= 1
#         if length(sorted_envelopes) == 1
#             append!(final_envelopes, sorted_envelopes)
#             popfirst!(sorted_envelopes)
#         else
#             envelope = first(sorted_envelopes)
#             to_compare_list = sorted_envelopes[2:end]
#             is_excluded = map(to_compare -> any(to_compare .∉ [envelope]), to_compare_list)
#             sorted_envelopes = to_compare_list[is_excluded]
#             push!(final_envelopes, envelope)
#         end
#     end
#     return final_envelopes
# end