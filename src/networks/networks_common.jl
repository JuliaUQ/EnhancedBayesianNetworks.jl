iscyclic(net::AbstractNetwork) = iscyclic(net.A)

isconnected(net::AbstractNetwork) = isconnected(net.A)

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
    push!(net.nodes, node)
    net.topology[node.name] = length(net.nodes)
    n = size(net.A, 1)
    Anew = spzeros(Bool, n + 1, n + 1)
    Anew[1:n, 1:n] = net.A
    net.A = Anew
end

function add_child!(
    net::Union{BayesianNetwork,CredalNetwork},
    par::Union{DiscreteNode,Vector{DiscreteNode}},
    ch::Union{DiscreteNode,Vector{DiscreteNode}}
)
    parents = wrap(par)
    children = wrap(ch)
    all_nodes = vcat(parents, children)
    missing_nodes = setdiff([i.name for i in all_nodes], [i.name for i in net.nodes])
    if !isempty(missing_nodes)
        error("Invalid Network: nodes $missing_nodes are not defined in the network")
    end
    ## verify No loop
    loop = intersect(parents, children)
    if !isempty(loop)
        error("Invalid Network: node '$(getproperty.(loop, :name))' has a loop")
    end
    ## verify Discrete parent nodes
    map(dp -> verify_discrete(dp, children), parents)
    pidx = getindex.(Ref(net.topology), getfield.(parents, :name))
    cidx = getindex.(Ref(net.topology), getfield.(children, :name))
    net.A[pidx, cidx] .= true
end

function add_child!(
    net::Union{BayesianNetwork,CredalNetwork},
    par::Union{Symbol,Vector{Symbol}},
    ch::Union{Symbol,Vector{Symbol}}
)
    parents = wrap(par)
    children = wrap(ch)
    all_nodes = vcat(parents, children)
    missing_nodes = setdiff(all_nodes, [i.name for i in net.nodes])
    if !isempty(missing_nodes)
        error("Invalid Network: nodes $missing_nodes are not defined in the network")
    end
    par_nodes = filter(x -> x.name ∈ parents, net.nodes)
    ch_nodes = filter(x -> x.name ∈ children, net.nodes)
    add_child!(net, par_nodes, ch_nodes)
end