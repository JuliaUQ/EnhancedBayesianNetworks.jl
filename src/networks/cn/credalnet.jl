"""
    CredalNetwork(nodes::AbstractVector{DiscreteNode})

A credal network: a DAG of discrete nodes whose CPTs may be **imprecise**
(interval-valued). Same layout as [`BayesianNetwork`](@ref) (`nodes`, `topology`,
`A`).

Validates that node names and states are globally unique; warns if every node is
precise, since a [`BayesianNetwork`](@ref) is the better fit in that case.
"""
mutable struct CredalNetwork <: AbstractNetwork
    nodes::AbstractVector{DiscreteNode}
    topology::Dict
    A::SparseMatrixCSC

    function CredalNetwork(nodes::AbstractVector{DiscreteNode}, topology::Dict, A::SparseMatrixCSC)
        # node names must be unique
        node_names = map(i -> i.name, nodes)
        dups = not_unique_elements(node_names)
        if !isempty(dups)
            error("Invalid CN: duplicate node names $dups")
        end
        # states must be globally unique across nodes (init=Symbol[] handles the empty-network case)
        states_list = reduce(vcat, states.(nodes); init=Symbol[])
        dups = not_unique_elements(states_list)
        if !isempty(dups)
            error("Invalid CN: duplicate node states $dups")
        end
        # a CredalNetwork targets imprecise nodes; warn if none are imprecise
        imprecise_nodes = nodes[.!isprecise.(nodes)]
        if isempty(imprecise_nodes)
            @warn "All the nodes are precise; BayesianNetwork structure should be used instead"
        end
        new(nodes, topology, A)
    end
end

function CredalNetwork(nodes::AbstractVector{DiscreteNode})
    n = length(nodes)
    topology = Dict{Symbol,Int}()
    for (i, n) in enumerate(nodes)
        topology[n.name] = i
    end
    A = spzeros(Bool, n, n)
    return CredalNetwork(nodes, topology, A)
end