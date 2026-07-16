"""
    EnhancedBayesianNetwork(nodes::AbstractVector{<:AbstractNode})

An enhanced Bayesian network: the modelling front-end that may mix discrete,
continuous, and functional nodes. Same layout as [`BayesianNetwork`](@ref)
(`nodes`, `topology`, `A`).

Validates that node names are unique and that states across discrete nodes are
globally unique. It is progressively transformed — via `discretize!`,
`transfer_continuous_functional_node!`, and [`reduce`](@ref) — into a
[`BayesianNetwork`](@ref) or [`CredalNetwork`](@ref) for inference.

# Examples
```julia
# a network mixing a discrete, a continuous, and a functional node:
W = DiscreteNode(:W, [:sunny => [Parameter(1.0, :W)], :cloudy => [Parameter(2.0, :W)]])
W[:W => :sunny] = 0.5; W[:W => :cloudy] = 0.5
X = ContinuousNode(:X, Uniform(-1, 1), ExactDiscretization([-1.0, 0.0, 1.0]))
model = Model(df -> df.X .+ df.W, :Y)
F = DiscreteFunctionalNode(:F, [model], df -> df.Y, MonteCarlo(200))

ebn = EnhancedBayesianNetwork([W, X, F])
add_child!(ebn, :W, :F); add_child!(ebn, :X, :F); order!(ebn)
# reduce(ebn) then turns it into a BayesianNetwork / CredalNetwork for inference
```
"""
mutable struct EnhancedBayesianNetwork <: AbstractNetwork
    nodes::AbstractVector{<:AbstractNode}
    topology::Dict
    A::SparseMatrixCSC

    function EnhancedBayesianNetwork(
        nodes::AbstractVector{<:AbstractNode},
        topology::Dict,
        A::SparseMatrixCSC
    )
        # node names must be unique
        node_names = map(i -> i.name, nodes)
        dups = _not_unique_elements(node_names)
        if !isempty(dups)
            error("Invalid eBN: duplicate node names $dups")
        end
        discrete_nodes = filter(x -> isa(x, DiscreteNode), nodes)
        # states must be globally unique across nodes (init=Symbol[] handles the empty-network case)
        states_list = reduce(vcat, states.(discrete_nodes); init=Symbol[])
        dups = _not_unique_elements(states_list)
        if !isempty(discrete_nodes)
            if !isempty(dups)
                error("Invalid eBN: duplicate node states $dups")
            end
        end
        new(nodes, topology, A)
    end
end

EnhancedBayesianNetwork(nodes::AbstractVector{<:AbstractNode}) = EnhancedBayesianNetwork(nodes, topology_and_adjacency(nodes)...)


"""
    markov_envelope(net::EnhancedBayesianNetwork)

Return the Markov envelopes of `net` as a vector of node-name vectors. Continuous nodes
linked through their Markov blankets are first collected into groups
([`markov_continuous_group`](@ref)); each group's envelope is the union of its members'
Markov blankets together with the members themselves. Envelopes that are a subset of
another envelope are discarded, so only the maximal (non-redundant) envelopes remain.

# Examples
```julia
W = DiscreteNode(:W, [:sunny => [Parameter(1.0, :W)], :cloudy => [Parameter(2.0, :W)]])
W[:W => :sunny] = 0.5; W[:W => :cloudy] = 0.5
X = ContinuousNode(:X, Uniform(-1, 1), ExactDiscretization([-1.0, 0.0, 1.0]))
F = DiscreteFunctionalNode(:F, [Model(df -> df.X .+ df.W, :Y)], df -> df.Y, MonteCarlo(200))
ebn = EnhancedBayesianNetwork([W, X, F])
add_child!(ebn, :W, :F); add_child!(ebn, :X, :F); order!(ebn)

markov_envelope(ebn)                        # [[:F, :W, :X]]  (continuous X, its child F, co-parent W)
```
"""
function markov_envelope(net::EnhancedBayesianNetwork)
    Xm_groups = map(n -> markov_continuous_group(net, n), filter(x -> isa(x, AbstractContinuousNode), net.nodes))
    envelopes = Vector{Vector{Symbol}}()
    for Xm_group in Xm_groups
        envelope = unique!(vcat(map(n -> vcat(markov_blanket(net, n), n.name), Xm_group)...))
        push!(envelopes, envelope)
    end

    sets = unique(Set.(envelopes))
    envelopes = collect.(sets)

    keep = trues(length(sets))
    for i in eachindex(sets)
        if any(j -> j != i && issubset(sets[i], sets[j]), eachindex(sets))
            keep[i] = false
        end
    end

    return envelopes[keep]
end

# Replace every continuous node that carries a discretization with a discrete surrogate node (the
# per-interval probability masses) plus a residual continuous node, rewiring the parents to the
# discrete part and the children to the continuous part.
function discretize!(net::EnhancedBayesianNetwork)
    continuous_nodes = filter(x -> isa(x, ContinuousNode), net.nodes)
    evidence_nodes = filter(n -> !isempty(n.discretization), continuous_nodes)
    discretization_tuples = map(n -> (n, parents(net, n), children(net, n), _discretize(n)), evidence_nodes)
    for (node, pars, chs, (discretized_node, new_continuous)) in discretization_tuples
        remove_node!(net, node)
        push_node!(net, discretized_node)
        push_node!(net, new_continuous)
        add_child!(net, discretized_node, new_continuous)
        map(p -> add_child!(net, p, discretized_node.name), pars)
        map(c -> add_child!(net, new_continuous.name, c), chs)
    end
end

# A continuous functional node with no discretization but with children is not evaluated on its own:
# prepend its models to each child's models and splice it out, linking its parents straight to its children.
function transfer_continuous_functional_node!(net::EnhancedBayesianNetwork, node::ContinuousFunctionalNode)
    node_children = filter(n -> n.name ∈ children(net, node), net.nodes)
    if isempty(node.discretization) && !isempty(node_children)
        node_parents = filter(n -> n.name ∈ parents(net, node), net.nodes)
        map(ch -> prepend!(ch.models, node.models), node_children)
        remove_node!(net, node)
        add_child!(net, node_parents, node_children)
        return order!(net)
    end
end

# Grow the set of continuous nodes linked through shared Markov blankets: keep adding continuous nodes
# found in the current group's blankets until it stabilises. Used to build Markov envelopes.
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

# Verify a functional node's parents: every discrete parent must carry non-empty parameters (they feed
# the models). Warns when there are no continuous parents (failure probabilities become Boolean) or no
# discrete ancestors (the network is a plain reliability analysis).
function _verify_functional_parents(net::EnhancedBayesianNetwork, node::FunctionalNode) ## Discrete Parents must have a non empty parameters attribute
    par = filter(n -> n.name ∈ parents(net, node), net.nodes)
    discrete_par = filter(x -> isa(x, AbstractDiscreteNode), par)
    cont_par = filter(x -> isa(x, AbstractContinuousNode), par)

    for dp in discrete_par
        if isempty(dp.parameters)
            error("Invalid Network: node $(repr(dp.name)) is a parent for the FunctionalNode $(repr(node.name)) and cannot have an empty parameters attribute")
        end
    end
    if isempty(cont_par)
        @warn "Node $(repr(node.name)) is a FunctionalNode with no continuous parents. Resulting failure probabilities are Boolean"
    end
    if isempty(discrete_ancestors(net, node))
        @warn "Node $(repr(node.name)) is a FunctionalNode with no discrete parents. Resulting network is a standard reliability analysis"
    end
end

# Materialise a functional node's per-scenario simulation table: for every combination of its discrete
# ancestors' states, store the node's simulation strategy. No-op if the table already exists.
function build_simulations!(net::EnhancedBayesianNetwork, node::FunctionalNode)
    if !isa(node.simulation, ScenariosTable)
        anc_nodes = filter(n -> n.name ∈ discrete_ancestors(net, node), net.nodes)
        anc = Symbol[i.name for i in anc_nodes]
        if isa(node, AbstractContinuousNode)
            st = ScenariosTable{ContinuousSimulation}(anc, :sim)
        else
            st = ScenariosTable{DiscreteSimulation}(anc, :sim)
        end
        theoretical_scenarios = vec(collect(Iterators.product(states.(anc_nodes)...)))
        map(th_s -> st[(anc .=> th_s)...] = node.simulation, theoretical_scenarios)
        node.simulation = st
    end
end

# Verify that the ancestors in a functional node's simulation table match its discrete ancestors in the
# eBN exactly — none defined only in the table, none missing from it.
function _verify_ancestors(net::EnhancedBayesianNetwork, node::FunctionalNode) ## verify if all the ancestors in the ST have been added via add_child!
    st_ancestors = Symbol.(names(node.simulation.data[:, Not(:sim)]))
    net_ancestors = discrete_ancestors(net, node)
    only_in_st = setdiff(st_ancestors, net_ancestors)
    if !isempty(only_in_st)
        error("Invalid SimulationTable: node $(repr(node.name)) has nodes $only_in_st defined in the SimulationTable only, but they are not ancestors in the defined eBN")
    end
    only_in_net = setdiff(net_ancestors, st_ancestors)
    if !isempty(only_in_net)
        error("Invalid SimulationTable: node $(repr(node.name)) has ancestors $only_in_net defined in the eBN only, but they are not present in its SimulationTable")
    end
end

# Every combination of the functional node's discrete-ancestor states must appear as a row in its simulation table.
function _verify_scenarios(net::EnhancedBayesianNetwork, node::FunctionalNode)
    anc = filter(n -> n.name ∈ discrete_ancestors(net, node), net.nodes)
    cols = [i.name for i in anc]
    present = Set(Tuple(r[c] for c in cols) for r in eachrow(node.simulation.data))
    for scenario in Iterators.product(states.(anc)...)
        if scenario ∉ present
            filtering_element = cols .=> scenario
            error("Invalid SimulationTable: node $(repr(node.name)) is missing the following scenario $(filtering_element)")
        end
    end
end