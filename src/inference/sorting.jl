function sort_nodes(bn::BayesianNetwork, scorefun)

    ig = InteractionGraph(bn)
    ns = NetworkSchema(bn)
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

function best_node(ig::InteractionGraph, ns::NetworkSchema, remaining::Set{Int}, scorefun)
    best = minimum(remaining)
    best_score = scorefun(ig, ns, best)

    for node in remaining
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

function fill_score(ig::InteractionGraph, _::NetworkSchema, node::Int)
    ed = deleted_edges(ig, node)
    if ed == 0
        return 0.0
    end
    ea = added_edges(ig, node)
    return ea / ed
end

function factor_score(ig::InteractionGraph, ns::NetworkSchema, node::Int)
    score = length(ns.idx_to_state[node])
    for neigh in ig.neighbors[node]
        score *= length(ns.idx_to_state[neigh])
    end
    return score
end

function fill_factor_score(ig::InteractionGraph, ns::NetworkSchema, node::Int)
    return (
        fill_score(ig, ns, node),
        factor_score(ig, ns, node),
        node
    )
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

