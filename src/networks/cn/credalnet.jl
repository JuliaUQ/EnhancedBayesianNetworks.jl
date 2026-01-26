@auto_hash_equals mutable struct CredalNetwork <: AbstractNetwork
    nodes::AbstractVector{<:AbstractNode}
    topology::Dict
    A::SparseMatrixCSC

    function CredalNetwork(nodes::AbstractVector{<:AbstractNode}, topology::Dict, A::SparseMatrixCSC)
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
        continuous_nodes = nodes[isa.(nodes, ContinuousNode)]
        continuous_nodes_names = [i.name for i in continuous_nodes]
        if !isempty(continuous_nodes)
            error("node/s $continuous_nodes_names are continuous. Use EnhancedBayesianNetwork structure!")
        end
        imprecise_nodes = nodes[map(!, isprecise.(nodes))]
        if isempty(imprecise_nodes)
            error("all nodes are precise. Use BayesianNetwork structure!")
        end
        new(nodes, topology, A)
    end
end

function CredalNetwork(nodes::AbstractVector{<:AbstractNode})
    n = length(nodes)
    topology = Dict()
    for (i, n) in enumerate(nodes)
        topology[n.name] = i
    end
    A = sparse(zeros(n, n))
    return CredalNetwork(nodes, topology, A)
end

function CredalNetwork(net::EnhancedBayesianNetwork)
    order!(net)
    nodes = net.nodes
    topology = net.topology
    A = net.A
    return CredalNetwork(nodes, topology, A)
end