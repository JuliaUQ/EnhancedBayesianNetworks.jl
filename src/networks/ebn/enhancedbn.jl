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
    all_nodes = vcat(parents, children)
    missing_nodes = setdiff([i.name for i in all_nodes], [i.name for i in net.nodes])
    if !isempty(missing_nodes)
        error("node(s) $missing_nodes is (are) not defined in the eBN")
    end
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
    net::EnhancedBayesianNetwork,
    par::Union{Symbol,Vector{Symbol}},
    ch::Union{Symbol,Vector{Symbol}}
)
    parents = wrap(par)
    children = wrap(ch)
    all_nodes = vcat(parents, children)
    missing_nodes = setdiff(all_nodes, [i.name for i in net.nodes])
    if !isempty(missing_nodes)
        error("node(s) $missing_nodes is (are) not defined in the eBN")
    end
    par_nodes = filter(x -> x.name ∈ parents, net.nodes)
    ch_nodes = filter(x -> x.name ∈ children, net.nodes)
    add_child!(net, par_nodes, ch_nodes)
end

function order!(net::EnhancedBayesianNetwork)
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
    rev = Dict(v => k for (k, v) in net.topology)
    ordered_topology = Dict(map(i -> (rev[i[2]], i[1]), enumerate(ordered_indices)))
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
    map(n -> build_simulation_table!(net, n), filter(x -> isa(x, FunctionalNode), net.nodes))
    map(n -> verify_ancestors(net, n), filter(x -> isa(x, FunctionalNode), net.nodes))
    map(n -> verify_scenarios(net, n), filter(x -> isa(x, FunctionalNode), net.nodes))

    return nothing
end

function markov_envelope(net::EnhancedBayesianNetwork)
    Xm_groups = map(n -> markov_continuous_group(net, n), filter(x -> isa(x, AbstractContinuousNode), net.nodes))
    envelopes = Vector{Vector{Symbol}}()
    for Xm_group in Xm_groups
        envelope = unique!(vcat(map(n -> vcat(markov_blanket(net, n), n.name), Xm_group)...))
        push!(envelopes, envelope)
    end

    sets = unique(Set.(envelopes))
    envelopes = collect.(sets)
    keep = trues(length(envelopes))
    for i in eachindex(sets), j in eachindex(sets)
        if i != j && issubset(sets[i], sets[j])
            keep[i] = false
            break
        end
    end

    return envelopes[keep]
end

function markov_continuous_group(net::EnhancedBayesianNetwork, node::Union{ContinuousNode,ContinuousFunctionalNode})
    Xm_group = [node]
    blanket = filter(n -> n.name ∈ markov_blanket(net, node), net.nodes)
    continuous_node_in_blanket = filter(x -> isa(x, AbstractContinuousNode), blanket)
    Xm_group_new = unique!(vcat(continuous_node_in_blanket, Xm_group))

    while !issetequal(Xm_group, Xm_group_new)
        new_nodes = setdiff(Xm_group_new, Xm_group)
        Xm_group = Xm_group_new
        blankets = unique!(vcat(map(n -> markov_blanket(net, n), new_nodes)...))
        blankets = filter(n -> n.name in blankets, net.nodes)
        continuous_node_in_blanket = filter(x -> isa(x, AbstractContinuousNode), blankets)
        Xm_group_new = unique!(vcat(continuous_node_in_blanket, Xm_group))
    end

    return Xm_group_new
end

function verify_parents(net::EnhancedBayesianNetwork, node::ContinuousNode) ## verify if all the parents in the CPT have been added via add_child!
    cpt_parents = parents(node)
    net_parents = parents(net, node.name)
    only_in_cpt = setdiff(cpt_parents, net_parents)
    if !isempty(only_in_cpt)
        error("Invalid CPT: node $(node.name) has node(s) '$only_in_cpt' defined in the CPT only, but they have not been added via add_child!")
    end
end

function verify_parents(_::EnhancedBayesianNetwork, _::FunctionalNode) ## verify if all the parents in the CPT have been added via add_child!
    return nothing
end

function verify_functional_parents(net::EnhancedBayesianNetwork, node::FunctionalNode) ## Discrete Parents must have a non empty parameters attribute
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

function build_simulation_table!(net::EnhancedBayesianNetwork, node::FunctionalNode)
    if isa(node.simulation, SimulationTable)
        return nothing
    else
        anc = Symbol[]
        anc_nodes = filter(n -> n.name ∈ ancestors(net, node), net.nodes)
        append!(anc, [i.name for i in anc_nodes])
        # anc = filter(n -> n.name ∈ ancestors(net, node), net.nodes)
        if isa(node, AbstractContinuousNode)
            st = SimulationTable{ContinuousSimulation}(anc)
        else
            st = SimulationTable{DiscreteSimulation}(anc)
        end
        theoretical_scenarios = vec(collect(Iterators.product(states.(anc_nodes)...)))
        map(th_s -> st[([i.name for i in anc_nodes] .=> th_s)...] = node.simulation, theoretical_scenarios)
        node.simulation = st
    end
end

function verify_ancestors(net::EnhancedBayesianNetwork, node::FunctionalNode) ## verify if all the ancestors in the ST have been added via add_child!
    st_ancestors = Symbol.(names(node.simulation.data[:, Not(:sim)]))
    net_ancestors = ancestors(net, node)
    only_in_st = setdiff(st_ancestors, net_ancestors)
    if !isempty(only_in_st)
        error("Invalid SimulationTable: node $(node.name) has node(s) '$only_in_st' defined in the SimulationTable only, but they are not ancestor(s) in the defined eBN")
    end
    only_in_net = setdiff(net_ancestors, st_ancestors)
    if !isempty(only_in_net)
        error("Invalid SimulationTable: node $(node.name) has ancestors(s) '$only_in_net' defined in the eBN only, but they are not present in its SimulationTable")
    end
end

function verify_scenarios(net::EnhancedBayesianNetwork, node::FunctionalNode)
    anc = filter(n -> n.name ∈ ancestors(net, node), net.nodes)
    theoretical_scenarios = vec(collect(Iterators.product(states.(anc)...)))
    filtering_elements = map(th_s -> ([i.name for i in anc] .=> th_s), theoretical_scenarios)
    for filtering_element in filtering_elements
        if isempty(filter(node.simulation, filtering_element...))
            error("Invalid SimulationTable: node $(node.name) is missing the following scenario $(filtering_element)")
        end
    end
end