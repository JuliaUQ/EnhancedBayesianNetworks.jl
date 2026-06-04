mutable struct BayesianNetwork <: AbstractNetwork
    nodes::AbstractVector{DiscreteNode}
    topology::Dict
    A::SparseMatrixCSC

    function BayesianNetwork(nodes::AbstractVector{DiscreteNode}, topology::Dict, A::SparseMatrixCSC)
        node_names = map(i -> i.name, nodes)
        dups = not_unique_elements(node_names)
        if !isempty(dups)
            error("Invalid BN: duplicate node names $dups")
        end
        states_list = mapreduce(i -> states(i), vcat, nodes)
        dups = not_unique_elements(states_list)
        if !isempty(dups)
            error("Invalid BN: duplicate node states $dups")
        end
        imprecise_nodes = nodes[.!isprecise.(nodes)]
        if !isempty(imprecise_nodes)
            error("Invalid BN: node/s $(getproperty.(imprecise_nodes, :name)) are imprecise; CredalNetwork structure is required")
        end
        new(nodes, topology, A)
    end
end

function BayesianNetwork(nodes::AbstractVector{DiscreteNode})
    n = length(nodes)
    topology = Dict{Symbol,Int}()
    for (i, n) in enumerate(nodes)
        topology[n.name] = i
    end
    A = spzeros(Bool, n, n)
    return BayesianNetwork(nodes, topology, A)
end

function joint_probability(bn::BayesianNetwork, scenario::Evidence)
    missing_names = setdiff(getproperty.(bn.nodes, :name), keys(scenario))

    if !isempty(missing_names)
        error("Node(s) $missing_names are not defined in the scenario $scenario. Use Inference instead")
    end

    extra_names = setdiff(keys(scenario), getproperty.(bn.nodes, :name))
    if !isempty(extra_names)
        @warn("Defined scenario contains $extra_names that are not defined in the BN. Therefore is useless for the scenario probability evaluation")
        for k in extra_names
            delete!(scenario, k)
        end
    end

    for (n, s) in scenario
        sts = states(first(filter(x -> x.name == n, bn.nodes)))
        if s ∉ sts
            error("Scenario defined state $s for node $n that does not belongs to its possible states $sts")
        end
    end

    prob = 1.0
    for node in bn.nodes
        query = vcat(node.name, parents(node))
        filtered = filter(p -> first(p) in query, scenario)
        prob_n = node.cpt[filtered...]
        prob *= prob_n
    end
    return prob
end