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

function query_to_idx(query::Vector{Symbol}, ns::NetworkSchema)
    return [ns.node_to_idx[q] for q in query]
end

function evidence_to_idx(evidence::Evidence, ns::NetworkSchema)
    result = Tuple{Int,Int}[]
    for (node, state) in evidence
        nodeid = ns.node_to_idx[node]
        stateid =
            ns.state_to_idx[nodeid][state]
        push!(result, (nodeid, stateid))
    end
    return result
end