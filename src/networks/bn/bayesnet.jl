@auto_hash_equals mutable struct BayesianNetwork <: AbstractNetwork
    nodes::AbstractVector{<:AbstractNode}
    topology::Dict
    A::SparseMatrixCSC

    function BayesianNetwork(nodes::AbstractVector{<:AbstractNode}, topology::Dict, A::SparseMatrixCSC)
        nodes_names = map(i -> i.name, nodes)
        if !allunique(nodes_names)
            error("network nodes names must be unique")
        end
        discrete_nodes = filter(x -> isa(x, DiscreteNode), nodes)
        if !isempty(discrete_nodes)
            states_list = mapreduce(i -> states(i), vcat, discrete_nodes)
            if !allunique(states_list)
                error("network nodes states must be unique")
            end
        end
        functional_nodes = nodes[isa.(nodes, FunctionalNode)]
        functional_nodes_names = [i.name for i in functional_nodes]
        if !isempty(functional_nodes)
            error("node/s $functional_nodes_names are functional nodes. evaluate the related EnhancedBayesianNetwork structure before!")
        end
        continuous_nodes = nodes[isa.(nodes, ContinuousNode)]
        continuous_nodes_names = [i.name for i in continuous_nodes]
        if !isempty(continuous_nodes)
            error("node/s $continuous_nodes_names are continuous. Use EnhancedBayesianNetwork structure!")
        end
        imprecise_nodes = nodes[map(!, isprecise.(nodes))]
        imprecise_nodes_names = [i.name for i in imprecise_nodes]
        if !isempty(imprecise_nodes)
            error("node/s $imprecise_nodes_names are imprecise. Use CrealNetwork structure!")
        end
        new(nodes, topology, A)
    end
end

function BayesianNetwork(nodes::AbstractVector{<:AbstractNode})
    n = length(nodes)
    topology = Dict()
    for (i, n) in enumerate(nodes)
        topology[n.name] = i
    end
    A = sparse(zeros(n, n))
    return BayesianNetwork(nodes, topology, A)
end

function BayesianNetwork(net::EnhancedBayesianNetwork)
    order!(net)
    nodes = net.nodes
    topology = net.topology
    A = net.A
    return BayesianNetwork(nodes, topology, A)
end

function joint_probability(bn::BayesianNetwork, scenario::Evidence)
    th_keys = [i.name for i in bn.nodes]
    pr_keys = keys(scenario) |> collect
    if !issubset(th_keys, pr_keys)
        error("Not all the BN's nodes $([i.name for i in bn.nodes]) have a specidied states in $scenario. Use Inference!")
    end
    for k in setdiff(pr_keys, th_keys)
        @warn("nodes $k is not part of the BN, therefore is useless for the scenario probability evaluation")
        delete!(scenario, k)
    end
    th_states = Dict(map(n -> (n.name, states(n)), bn.nodes))
    for (node, th_state) in th_states
        if scenario[node] ∉ th_state
            error("node $node has a defined scenario state $(scenario[node]) that is not among its possible states $th_state")
        end
    end

    prob = 1.0
    cpts_dict = Dict(map(n -> (n.name, n.cpt.data), bn.nodes))
    parents_dict = Dict(map(n -> (n.name, parents(bn, n)[2]), bn.nodes))
    for (node, cpt) in cpts_dict
        parent_keys = get(parents_dict, node, [])
        all_keys = vcat(parent_keys, node)
        row = filter(r -> all(k -> r[k] == scenario[k], all_keys), eachrow(cpt))
        prob *= row.Π[1]
    end
    return prob
end