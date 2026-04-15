struct InferenceState <: AbstractInferenceState
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

function verify_query(query::Vector{Symbol}, net::Union{BayesianNetwork,CredalNetwork}, evidence::Evidence)
    missing_names = setdiff(query, getproperty.(net.nodes, :name))
    if !isempty(missing_names)
        error("Query $query contains Symbol(s) $missing_names that are names of the nodes of the network")
    end
    overlap = intersect(query, keys(evidence))
    if !isempty(overlap)
        error("Query $query contains Symbol(s) $overlap that are already part of the evidence $evidence")
    end
end

function verify_evidence(evidence::Evidence, net::Union{BayesianNetwork,CredalNetwork})
    missing_names = setdiff(keys(evidence), getproperty.(net.nodes, :name))
    if !isempty(missing_names)
        error("Evidence $evidence contains Symbol(s) $missing_names that are names of the nodes of the network")
    end
    for (n, s) in evidence
        sts = states(first(filter(x -> x.name == n, net.nodes)))
        if s ∉ sts
            error("Evidence defined state $s for node $n that does not belongs to its possible states $sts")
        end
    end
end