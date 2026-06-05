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
        error("Invalid Query: queried nodes vector $query contains Symbols $missing_names that are not associated to any node of the network")
    end
    overlap = intersect(query, keys(evidence))
    if !isempty(overlap)
        evidence_str = "[" * join(["$(repr(k)) => $(repr(v))" for (k, v) in evidence], ", ") * "]"
        error("Invalid Query: queried nodes vector $query contains Symbols $overlap that are already part of the evidence $evidence_str")
    end
end

function verify_evidence(evidence::Evidence, net::Union{BayesianNetwork,CredalNetwork})
    missing_names = setdiff(keys(evidence), getproperty.(net.nodes, :name))
    if !isempty(missing_names)
        evidence_str = "[" * join(["$(repr(k)) => $(repr(v))" for (k, v) in evidence], ", ") * "]"
        missing_str = string(collect(missing_names))
        error("Invalid Evidence: evidence $evidence_str contains Symbols $missing_str that are not associated to any node of the network")
    end
    for (n, s) in evidence
        evidence_str = "[" * join(["$(repr(k)) => $(repr(v))" for (k, v) in evidence], ", ") * "]"
        sts = states(first(filter(x -> x.name == n, net.nodes)))
        if s ∉ sts
            error("Invalid Evidence: evidence $evidence_str defines state $(repr(s)) for node $(repr(n)) that does not belong to its possible states $(repr(sts))")
        end
    end
end