struct NetworkSchema
    node_to_idx::Dict{Symbol,Int}
    idx_to_node::Vector{Symbol}
    state_to_idx::Vector{Dict{Symbol,Int}}
    idx_to_state::Vector{Vector{Symbol}}
end

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

struct InteractionGraph
    neighbors::Vector{Set{Int}}
end

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

struct InferenceState
    net::Union{BayesianNetwork,CredalNetwork}
    query::Vector{Symbol}
    evidence::Evidence

    function InferenceState(net::Union{BayesianNetwork,CredalNetwork}, query::Union{Symbol,Vector{Symbol}}, evidence::Evidence)
        query = wrap(query)
        verify_evidence(evidence, net)
        verify_query(query, net, evidence)
        return new(net, query, evidence)
    end
end

include("utils.jl")
include("factors.jl")
include("factors_algebra.jl")
include("sorting.jl")
include("variableselimination.jl")
include("infer.jl")