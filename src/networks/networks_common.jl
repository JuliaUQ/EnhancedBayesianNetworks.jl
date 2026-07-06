function order!(net::AbstractNetwork)
    if iscyclic(net.A)
        error("Invalid Network: network is cyclic!")
    end
    if !isconnected(net.A)
        error("Invalid Network: network is not connected")
    end
    topologically_sort!(net)
    foreach(n -> verify_parents(net, n), net.nodes)
    foreach(filter(x -> isa(x, DiscreteNode), net.nodes)) do n
        verify_scenarios(net, n)
    end
    foreach(filter(x -> isa(x, DiscreteNode), net.nodes)) do n
        verify_exhaustiveness(net, n)
    end
end

# add edges from node objects; discrete/continuous/functional parents are each verified
# against the child type (the continuous/functional filters are empty for BN/CN)
function add_child!(
    net::AbstractNetwork,
    par::Union{<:AbstractNode,Vector{<:AbstractNode}},
    ch::Union{<:AbstractNode,Vector{<:AbstractNode}}
)
    parents = wrap(par)
    children = wrap(ch)
    assert_nodes_defined(net, [i.name for i in vcat(parents, children)])
    loop = intersect(parents, children)
    isempty(loop) || error("Invalid Network: nodes $(getproperty.(loop, :name)) have a loop")
    map(dp -> verify_discrete(dp, children), filter(x -> isa(x, DiscreteNode), parents))
    map(cfp -> verify_continuous_and_functional(cfp, children), filter(x -> isa(x, Union{ContinuousNode,FunctionalNode}), parents))
    set_edges!(net, parents, children)
end

# add edges by node name
function add_child!(
    net::AbstractNetwork,
    par::Union{Symbol,Vector{Symbol}},
    ch::Union{Symbol,Vector{Symbol}}
)
    parents = wrap(par)
    children = wrap(ch)
    assert_nodes_defined(net, vcat(parents, children))
    par_nodes = filter(x -> x.name ∈ parents, net.nodes)
    ch_nodes = filter(x -> x.name ∈ children, net.nodes)
    add_child!(net, par_nodes, ch_nodes)
end

function parents(net::AbstractNetwork, name::Symbol)
    parents_idx = findnz(net.A[:, net.topology[name]])[1]
    return Symbol[net.nodes[idx].name for idx in parents_idx]
end

parents(net::AbstractNetwork, node::AbstractNode) = parents(net, node.name)

function children(net::AbstractNetwork, name::Symbol)
    children_idx = findnz(net.A[net.topology[name], :])[1]
    return Symbol[net.nodes[idx].name for idx in children_idx]
end

children(net::AbstractNetwork, node::AbstractNode) = children(net, node.name)

"""
    markov_blanket(net::AbstractNetwork, node)

Return the Markov blanket of `node` (given by name as a `Symbol` or as a node object) as
a vector of node names: its parents, its children, and its children's other parents
(co-parents). The node itself is excluded. Conditioned on its Markov blanket, a node is
conditionally independent of every other node in the network.
"""
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

iscyclic(net::AbstractNetwork) = iscyclic(net.A)

isconnected(net::AbstractNetwork) = isconnected(net.A)

function discrete_ancestors(net::AbstractNetwork, name::Symbol)
    start_idx = net.topology[name]
    visited = Set{Int}()
    result = Set{Symbol}()
    stack = [start_idx]
    while !isempty(stack)
        current = pop!(stack)
        for p in findnz(net.A[:, current])[1]
            if p ∉ visited
                push!(visited, p)
                node = net.nodes[p]
                if node isa AbstractDiscreteNode
                    push!(result, node.name)
                else
                    push!(stack, p)
                end
            end
        end
    end
    return Symbol[n for n in result]
end

discrete_ancestors(net::AbstractNetwork, node::AbstractNode) = discrete_ancestors(net, node.name)

function topologically_sort!(net::AbstractNetwork)
    order = topologically_sort(net.A)
    net.nodes = net.nodes[order]
    net.A = net.A[order, order]
    for (i, node) in enumerate(net.nodes)
        net.topology[node.name] = i
    end
end

function verify_parents(_::AbstractNetwork, _::AbstractNode) ## verify if all the parents in the CPT have been added via add_child!
    return
end

function verify_parents(net::AbstractNetwork, node::Union{DiscreteNode,ContinuousNode}) ## verify if all the parents in the CPT have been added via add_child!
    cpt_parents = parents(node)
    net_parents = parents(net, node.name)
    only_in_cpt = setdiff(cpt_parents, net_parents)
    if !isempty(only_in_cpt)
        error("Invalid CPT: node $(repr(node.name)) has nodes $only_in_cpt defined in the CPT only, but they have not been added via add_child!")
    end
end

# Every combination of (parent states × own states) must appear in the CPT. Build the set of present rows once, then membership-test each theoretical scenario
function verify_scenarios(net::AbstractNetwork, node::DiscreteNode)
    par = filter(n -> n.name ∈ parents(node), net.nodes)
    v = vcat(par, node)
    cols = [n.name for n in v]
    present = Set(Tuple(r) for r in eachrow(node.cpt.data[:, cols]))
    for scenario in Iterators.product(states.(v)...)
        if scenario ∉ present
            filtering_element = cols .=> scenario
            error("Invalid CPT: node $(repr(node.name)) is missing the following scenario $(filtering_element)")
        end
    end
end

# For each parent-state combination the own-state probabilities must be exhaustive: sum ≈ 1 (precise) or the interval sum must bracket 1 (imprecise). 
# groupby partitions the CPT by parent columns in a single pass. Assumes every combination is present.
# order! runs verify_scenarios first, so a missing group cannot reach here.
function verify_exhaustiveness(net::AbstractNetwork, node::DiscreteNode)
    par = [n.name for n in filter(n -> n.name ∈ parents(node), net.nodes)]
    groups = isempty(par) ? (node.cpt.data,) : groupby(node.cpt.data, par)
    precise = isprecise(node)
    for sub in groups
        scenario = [p => sub[1, p] for p in par]
        if precise
            if !isapprox(sum(sub.Π), 1)
                valstr = "[" * join(string.(sub.Π), ", ") * "]"
                error("Invalid CPT: node $(repr(node.name)) has CPT values $valstr not exhaustive and mutually exclusive for the scenario $scenario")
            end
        else
            lb_sum, ub_sum = EnhancedBayesianNetworks.sum_intervals_and_float(sub.Π...)
            if lb_sum > 1
                valstr = "[" * join(string.(sub.Π), ", ") * "]"
                error("Invalid CPT: node $(repr(node.name)) has CPT values $valstr for the scenario $scenario, the sum of lower bound values must be less than 1")
            elseif ub_sum < 1
                valstr = "[" * join(string.(sub.Π), ", ") * "]"
                error("Invalid CPT: node $(repr(node.name)) has CPT values $valstr for the scenario $scenario, the sum of upper bound values must be greater than 1")
            end
        end
    end
end

function remove_node!(net::AbstractNetwork, node::AbstractNode)
    filter!(n -> n.name != node.name, net.nodes)
    idx = net.topology[node.name]
    keep = setdiff(1:size(net.A, 1), idx)
    net.A = net.A[keep, keep]
    delete!(net.topology, node.name)
    # shift indices
    for (k, v) in net.topology
        if v > idx
            net.topology[k] = v - 1
        end
    end
end

remove_node!(net::AbstractNetwork, name::Symbol) = remove_node!(net, first(filter(n -> n.name == name, net.nodes)))

function add_node!(net::AbstractNetwork, node::AbstractNode)
    if haskey(net.topology, node.name)
        error("Invalid Network: node $(repr(node.name)) is already present in the network")
    end
    push!(net.nodes, node)
    net.topology[node.name] = length(net.nodes)
    n = size(net.A, 1)
    Anew = spzeros(Bool, n + 1, n + 1)
    Anew[1:n, 1:n] = net.A
    net.A = Anew
end

# Build the (topology, adjacency) pair shared by all three network constructors:
# topology maps each node name to its 1-based position; A is the empty n×n edge matrix.
function topology_and_adjacency(nodes::AbstractVector{<:AbstractNode})
    topology = Dict{Symbol,Int}(node.name => i for (i, node) in enumerate(nodes))
    A = spzeros(Bool, length(nodes), length(nodes))
    return topology, A
end

# every referenced parent/child name must already be a node in the network
function assert_nodes_defined(net::AbstractNetwork, names::AbstractVector{Symbol})
    missing_nodes = setdiff(names, [i.name for i in net.nodes])
    isempty(missing_nodes) || error("Invalid Network: nodes $missing_nodes are not defined in the network")
end

# flip on the parent→child edges in the adjacency matrix
function set_edges!(net::AbstractNetwork, parents, children)
    pidx = getindex.(Ref(net.topology), getfield.(parents, :name))
    cidx = getindex.(Ref(net.topology), getfield.(children, :name))
    net.A[pidx, cidx] .= true
end