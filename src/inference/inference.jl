# Bidirectional name/state ↔ integer-id maps for a network, so inference can run on dense integer-
# indexed factors: node_to_idx/idx_to_node map node names ↔ ids; state_to_idx/idx_to_state map each
# node's state symbols ↔ their 1-based positions.
struct NetworkSchema
    node_to_idx::Dict{Symbol,Int}
    idx_to_node::Vector{Symbol}
    state_to_idx::Vector{Dict{Symbol,Int}}
    idx_to_state::Vector{Vector{Symbol}}
end

# Build the schema from a network: node ids come from bn.topology, each node's states from states(node).
function NetworkSchema(bn::BayesianNetwork)
    node_to_idx = copy(bn.topology)

    n = length(node_to_idx)
    idx_to_node = Vector{Symbol}(undef, n)
    for (node, idx) in node_to_idx
        idx_to_node[idx] = node
    end

    idx_to_state = Vector{Vector{Symbol}}(undef, n)
    for node in bn.nodes
        idx = node_to_idx[node.name]
        idx_to_state[idx] = states(node)
    end

    state_to_idx = [Dict(state => i for (i, state) in enumerate(sts)) for sts in idx_to_state]

    NetworkSchema(node_to_idx, idx_to_node, state_to_idx, idx_to_state)
end

# The moral graph of a network as adjacency sets: each node linked to its parents and children, and each
# node's parents "married" to one another (moral edges). Drives the elimination-ordering heuristics.
struct InteractionGraph
    neighbors::Vector{Set{Int}}
end

# Build the moral graph: for every node add its parent-child edges, then connect every pair of its parents.
function InteractionGraph(bn::BayesianNetwork)
    n = size(bn.A, 1)
    neighbors = [Set{Int}() for _ in 1:n]
    rows = rowvals(bn.A)
    for child in 1:n
        pars = rows[nzrange(bn.A, child)]
        # parent-child edges
        for p in pars
            push!(neighbors[child], p)
            push!(neighbors[p], child)
        end
        # moral edges
        for i in eachindex(pars)
            for j in (i+1):length(pars)
                p1 = pars[i]
                p2 = pars[j]
                push!(neighbors[p1], p2)
                push!(neighbors[p2], p1)
            end
        end
    end
    InteractionGraph(neighbors)
end

include("utils.jl")
include("factors.jl")
include("factors_algebra.jl")
include("sorting.jl")
include("variableselimination.jl")
include("infer.jl")