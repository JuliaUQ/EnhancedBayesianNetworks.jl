# Validate a query: every queried name must be a node in the network, and none may already be fixed by the evidence.
function verify_query(query::Union{Symbol,Vector{Symbol}}, net::Union{BayesianNetwork,CredalNetwork}, evidence::Evidence)
    query = _wrap(query)
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

# Validate evidence: every evidence name must be a node, and each assigned state must be one of that node's states.
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

# Map query node names to their integer ids in the schema.
function query_to_idx(query::Union{Symbol,Vector{Symbol}}, ns::NetworkSchema)
    query = _wrap(query)
    return [ns.node_to_idx[q] for q in query]
end

# Map evidence (name => state) pairs to (node id, state id) pairs in the schema.
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