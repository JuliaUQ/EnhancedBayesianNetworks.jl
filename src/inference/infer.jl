struct Posterior{T,A<:AbstractArray{T}}
    factor::Factor{T,A}
    schema::NetworkSchema
    query::Vector{Symbol}
    evidence::Evidence
end

function infer(
    bn::BayesianNetwork,
    query::Vector{Symbol},
    evidence::Evidence,
    scorefun=fill_factor_score
)

    ns = NetworkSchema(bn)
    ig = InteractionGraph(bn)
    factors = factorize(bn)

    query_vars = query_to_idx(query, ns)

    evidence_idx = evidence_to_idx(evidence, ns)

    order = sort_nodes(ig, ns, scorefun)

    result = ve(factors, order, query_vars, evidence_idx)

    return return Posterior(result, ns, query, evidence)
end