function sort_with_minimal_added_complexity_and_complexity(bn::BayesianNetwork, ns::NetworkSchema)
    sort_nodes(bn, (ig, node) -> (ic_score(ig, node), complexity_score(ig, node, ns), node))
end

function sort_with_minimal_added_complexity(bn::BayesianNetwork)
    sort_nodes(bn, (ig, node) -> ic_score(ig, node))
end

function sort_with_minimal_complexity(bn::BayesianNetwork, ns::NetworkSchema)
    sort_nodes(bn, (ig, node) -> complexity_score(ig, node, ns))
end

function sort_nodes(bn::BayesianNetwork, scorefun)

    ig = InteractionGraph(bn)
    remaining = Set(1:length(ig.neighbors))
    order = Int[]

    while !isempty(remaining)
        node = best_node(ig, remaining, scorefun)
        push!(order, node)
        delete!(remaining, node)
        eliminate!(ig, node)
    end

    return order
end

function best_node(ig::InteractionGraph, remaining::Set{Int}, scorefun)

    best_node = minimum(remaining)
    best_score = scorefun(ig, best_node)

    for node in remaining
        if node == best_node
            continue
        end
        score = scorefun(ig, node)
        if score < best_score
            best_score = score
            best_node = node
        end
    end

    return best_node
end

function ic_score(ig::InteractionGraph, node::Int)
    ed = deleted_edges(ig, node)
    if ed == 0
        return 0.0
    end
    ea = added_edges(ig, node)
    return ea / ed
end

function complexity_score(ig::InteractionGraph, node::Int, ns::NetworkSchema)
    score = length(ns.idx_to_state[node])
    for neigh in ig.neighbors[node]
        score *= length(ns.idx_to_state[neigh])
    end
    return score
end

function eliminate!(ig::InteractionGraph, node::Int)
    neigh = collect(ig.neighbors[node])
    # add fill-in edges
    for i in eachindex(neigh)
        for j in (i+1):length(neigh)
            n1 = neigh[i]
            n2 = neigh[j]
            push!(ig.neighbors[n1], n2)
            push!(ig.neighbors[n2], n1)
        end
    end
    # remove node from its neighbors
    for n in neigh
        delete!(ig.neighbors[n], node)
    end
    empty!(ig.neighbors[node])
    return ig
end

function added_edges(ig::InteractionGraph, node::Int)
    neigh = collect(ig.neighbors[node])
    missing = 0
    for i in eachindex(neigh)
        for j in (i+1):length(neigh)
            if !(neigh[j] in ig.neighbors[neigh[i]])
                missing += 1
            end
        end
    end
    return missing
end

@inline deleted_edges(ig::InteractionGraph, node::Int) = length(ig.neighbors[node])

