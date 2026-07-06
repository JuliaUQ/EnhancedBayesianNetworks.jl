"""
    BayesianNetwork(nodes::AbstractVector{DiscreteNode})

A Bayesian network: a DAG of discrete, **precise** nodes.

- `nodes`: the `DiscreteNode`s, positionally aligned with `topology`/`A`.
- `topology`: maps each node name to its index (row/column in `A`).
- `A`: sparse boolean adjacency matrix; `A[i, j] == true` iff `i → j`.

Validates that node names and states are globally unique and that every node is
precise — any imprecise node requires a [`CredalNetwork`](@ref). Edges are added
afterwards with [`add_child!`](@ref).
"""
mutable struct BayesianNetwork <: AbstractNetwork
    nodes::AbstractVector{DiscreteNode}
    topology::Dict
    A::SparseMatrixCSC

    function BayesianNetwork(nodes::AbstractVector{DiscreteNode}, topology::Dict, A::SparseMatrixCSC)
        # node names must be unique
        node_names = map(i -> i.name, nodes)
        dups = not_unique_elements(node_names)
        if !isempty(dups)
            error("Invalid BN: duplicate node names $dups")
        end
        # states must be globally unique across nodes (init=Symbol[] handles the empty-network case)
        states_list = reduce(vcat, states.(nodes); init=Symbol[])
        dups = not_unique_elements(states_list)
        if !isempty(dups)
            error("Invalid BN: duplicate node states $dups")
        end
        # a BayesianNetwork admits only precise nodes
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
    scenario = deepcopy(scenario)
    missing_names = setdiff(getproperty.(bn.nodes, :name), keys(scenario))
    if !isempty(missing_names)
        error("Invalid Scenario: nodes $missing_names are not defined in the scenario; joint_probability requires a complete scenario, use infer instead")
    end
    extra_names = setdiff(keys(scenario), getproperty.(bn.nodes, :name))
    if !isempty(extra_names)
        @warn("Scenario contains nodes $(collect(extra_names)) that are not defined in the network; they are ignored in the joint probability evaluation")
        for k in extra_names
            delete!(scenario, k)
        end
    end

    for (n, s) in scenario
        sts = states(first(filter(x -> x.name == n, bn.nodes)))
        if s ∉ sts
            error("Invalid Scenario: scenario defines state $(repr(s)) for node $(repr(n)) that does not belong to its possible states $(repr(sts))")
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

function sample(bn::BayesianNetwork, n::Int=1)
    order!(bn)
    evidences = [Evidence() for _ in 1:n]
    samples = DataFrame()
    for node in bn.nodes
        results = map(e -> sample(node, e), evidences)
        samples[!, node.name] = results
        # Update Evidence
        for (e, r) in zip(evidences, results)
            e[node.name] = r
        end
    end
    return samples
end