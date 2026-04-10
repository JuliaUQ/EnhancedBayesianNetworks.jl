mutable struct CredalNetwork <: AbstractNetwork
    nodes::AbstractVector{DiscreteNode}
    topology::Dict
    A::SparseMatrixCSC

    function CredalNetwork(nodes::AbstractVector{DiscreteNode}, topology::Dict, A::SparseMatrixCSC)
        node_names = map(i -> i.name, nodes)
        dups = not_unique_elements(node_names)
        if !isempty(dups)
            error("Invalid CN: duplicate node names $dups")
        end
        states_list = mapreduce(i -> states(i), vcat, nodes)
        dups = not_unique_elements(states_list)
        if !isempty(dups)
            error("Invalid CN: duplicate node states $dups")
        end
        imprecise_nodes = nodes[.!isprecise.(nodes)]
        if isempty(imprecise_nodes)
            @warn("All the nodes in the defined CN are precise. Use BayesianNetwork constructor instead")
        end
        new(nodes, topology, A)
    end
end

function CredalNetwork(nodes::AbstractVector{DiscreteNode})
    n = length(nodes)
    topology = Dict()
    for (i, n) in enumerate(nodes)
        topology[n.name] = i
    end
    A = spzeros(Bool, n, n)
    return CredalNetwork(nodes, topology, A)
end