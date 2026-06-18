struct NetworkSchema
    node_to_idx::Dict{Symbol,Int}
    idx_to_node::Vector{Symbol}
    state_to_idx::Vector{Dict{Symbol,Int}}
    idx_to_state::Vector{Vector{Symbol}}

    function NetworkSchema(bn::BayesianNetwork)
        idx_to_node = getproperty.(bn.nodes, :name)
        node_to_idx = Dict(
            v => i
            for (i, v) in enumerate(idx_to_node)
        )
        idx_to_state = states.(bn.nodes)
        state_to_idx = [
            Dict(state => i
                 for (i, state) in enumerate(sts))
            for sts in idx_to_state
        ]
        new(node_to_idx, idx_to_node, state_to_idx, idx_to_state)
    end
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
include("variableselimination.jl")