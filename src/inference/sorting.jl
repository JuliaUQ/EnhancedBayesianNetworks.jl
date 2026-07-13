"""
    fill_score(ig, ns, node)

Elimination-ordering heuristic (min-fill flavour): scores a node by the ratio of fill-in edges its
elimination would *add* to the edges it would *remove* (`0.0` when it has no neighbours). Lower scores
are eliminated earlier. Pass as the `scorefun` argument to [`infer`](@ref).

# Examples
```julia
W = DiscreteNode(:W); W[:W => :sunny] = 0.5; W[:W => :cloudy] = 0.5
S = DiscreteNode(:S, [:W])
S[:W => :sunny,  :S => :on] = 0.9; S[:W => :sunny,  :S => :off] = 0.1
S[:W => :cloudy, :S => :on] = 0.2; S[:W => :cloudy, :S => :off] = 0.8
bn = BayesianNetwork([W, S]); add_child!(bn, :W, :S); order!(bn)

infer(bn, :S, Evidence(:W => :sunny), fill_score)
```
"""
function fill_score(ig::InteractionGraph, _::NetworkSchema, node::Int)
    ed = deleted_edges(ig, node)
    if ed == 0
        return 0.0
    end
    ea = added_edges(ig, node)
    return ea / ed
end

"""
    factor_score(ig, ns, node)

Elimination-ordering heuristic (min-factor flavour): scores a node by the size of the factor its
elimination would create — the product of the state-space sizes of the node and its current neighbours.
Lower scores are eliminated earlier. Pass as the `scorefun` argument to [`infer`](@ref).

# Examples
```julia
W = DiscreteNode(:W); W[:W => :sunny] = 0.5; W[:W => :cloudy] = 0.5
S = DiscreteNode(:S, [:W])
S[:W => :sunny,  :S => :on] = 0.9; S[:W => :sunny,  :S => :off] = 0.1
S[:W => :cloudy, :S => :on] = 0.2; S[:W => :cloudy, :S => :off] = 0.8
bn = BayesianNetwork([W, S]); add_child!(bn, :W, :S); order!(bn)

infer(bn, :S, Evidence(:W => :sunny), factor_score)
```
"""
function factor_score(ig::InteractionGraph, ns::NetworkSchema, node::Int)
    score = length(ns.idx_to_state[node])
    for neigh ∈ ig.neighbors[node]
        score *= length(ns.idx_to_state[neigh])
    end
    return score
end

"""
    fill_factor_score(ig, ns, node)

Default elimination-ordering heuristic for [`infer`](@ref): a tuple `(fill_score, factor_score, node)`
compared lexicographically — break [`fill_score`](@ref) ties by the smaller resulting factor
([`factor_score`](@ref)), then by node id for determinism. Lower is eliminated earlier.

# Examples
```julia
W = DiscreteNode(:W); W[:W => :sunny] = 0.5; W[:W => :cloudy] = 0.5
S = DiscreteNode(:S, [:W])
S[:W => :sunny,  :S => :on] = 0.9; S[:W => :sunny,  :S => :off] = 0.1
S[:W => :cloudy, :S => :on] = 0.2; S[:W => :cloudy, :S => :off] = 0.8
bn = BayesianNetwork([W, S]); add_child!(bn, :W, :S); order!(bn)

infer(bn, :S, Evidence(:W => :sunny))                     # fill_factor_score is the default
infer(bn, :S, Evidence(:W => :sunny), fill_factor_score)  # or pass it explicitly
```
"""
function fill_factor_score(ig::InteractionGraph, ns::NetworkSchema, node::Int)
    return (
        fill_score(ig, ns, node),
        factor_score(ig, ns, node),
        node
    )
end

# Greedy elimination ordering: repeatedly pick the lowest-scoring remaining node, record it, and
# eliminate it from the interaction graph until none remain.
function sort_nodes(ig::InteractionGraph, ns::NetworkSchema, scorefun)
    remaining = Set(1:length(ig.neighbors))
    order = Int[]
    while !isempty(remaining)
        node = best_node(ig, ns, remaining, scorefun)
        push!(order, node)
        delete!(remaining, node)
        eliminate!(ig, node)
    end
    return order
end

# The remaining node with the smallest `scorefun` value (ties broken by whatever the score encodes).
function best_node(ig::InteractionGraph, ns::NetworkSchema, remaining::Set{Int}, scorefun)
    best = minimum(remaining)
    best_score = scorefun(ig, ns, best)
    for node ∈ remaining
        if node == best
            continue
        end
        score = scorefun(ig, ns, node)
        if score < best_score
            best_score = score
            best = node
        end
    end
    return best
end

# Eliminate a node from the moral graph: connect all its neighbours pairwise (fill-in edges), then
# remove the node from its neighbours and clear its own adjacency.
function eliminate!(ig::InteractionGraph, node::Int)
    neigh = collect(ig.neighbors[node])
    # add fill-in edges
    for i ∈ eachindex(neigh)
        for j ∈ (i+1):length(neigh)
            n1 = neigh[i]
            n2 = neigh[j]
            push!(ig.neighbors[n1], n2)
            push!(ig.neighbors[n2], n1)
        end
    end
    # remove node from its neighbors
    for n ∈ neigh
        delete!(ig.neighbors[n], node)
    end
    empty!(ig.neighbors[node])
    return ig
end

# Number of fill-in edges eliminating `node` would introduce (neighbour pairs not already adjacent).
function added_edges(ig::InteractionGraph, node::Int)
    neigh = collect(ig.neighbors[node])
    missing = 0
    for i ∈ eachindex(neigh)
        for j ∈ (i+1):length(neigh)
            if !(neigh[j] ∈ ig.neighbors[neigh[i]])
                missing += 1
            end
        end
    end
    return missing
end

# Edges removed by eliminating `node` = its current neighbour count.
@inline deleted_edges(ig::InteractionGraph, node::Int) = length(ig.neighbors[node])

