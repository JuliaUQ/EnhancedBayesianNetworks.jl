"""
    CredalNetwork(nodes::AbstractVector{DiscreteNode})

A credal network: a DAG of discrete nodes whose CPTs may be **imprecise**
(interval-valued). Same layout as [`BayesianNetwork`](@ref) (`nodes`, `topology`,
`A`).

Validates that node names and states are globally unique; warns if every node is
precise, since a [`BayesianNetwork`](@ref) is the better fit in that case.

# Examples
```julia
W = DiscreteNode(:W); W[:W => :sunny] = 0.5; W[:W => :cloudy] = 0.5
S = DiscreteNode(:S, [:W])
# imprecise (interval-valued) entries make this a credal, not a Bayesian, network:
S[:W => :sunny,  :S => :on]  = Interval(0.8, 0.95); S[:W => :sunny,  :S => :off] = Interval(0.05, 0.2)
S[:W => :cloudy, :S => :on]  = 0.2;                 S[:W => :cloudy, :S => :off] = 0.8

cn = CredalNetwork([W, S])
add_child!(cn, :W, :S); order!(cn)
```
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

CredalNetwork(nodes::AbstractVector{<:AbstractNode}) = CredalNetwork(nodes, topology_and_adjacency(nodes)...)
