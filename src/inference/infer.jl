struct Posterior{T,A<:AbstractArray{T}}
    factor::Factor{T,A}
    schema::NetworkSchema
    query::Vector{Symbol}
    evidence::Evidence
end

struct CredalPosterior{T,A<:AbstractArray{T}}
    posteriors::Vector{<:Posterior}
    lower::Factor{T,A}
    upper::Factor{T,A}
    schema::NetworkSchema
    query::Vector{Symbol}
    evidence::Evidence
end

function infer(
    bn::BayesianNetwork,
    query::Union{Symbol,Vector{Symbol}},
    evidence::Evidence,
    scorefun=fill_factor_score
)
    ns = NetworkSchema(bn)
    ig = InteractionGraph(bn)
    factors = factorize(bn)
    query = wrap(query)
    query_vars = query_to_idx(query, ns)
    evidence_idx = evidence_to_idx(evidence, ns)
    order = sort_nodes(ig, ns, scorefun)
    result = ve(factors, order, query_vars, evidence_idx)
    return Posterior(result, ns, query, evidence)
end

function infer(
    cn::CredalNetwork,
    query::Vector{Symbol},
    evidence::Evidence,
    scorefun=fill_factor_score
)
    posteriors = Posterior[]
    bns = extreme_bayesian_networks(cn)
    @info("Perfoming inference over $(length(collect(bns))) BNs")
    for bn in bns
        push!(posteriors, infer(bn, query, evidence, scorefun))
    end
    factors = getproperty.(posteriors, :factor)
    tables = getproperty.(factors, :table)

    lower_table = reduce((a, b) -> min.(a, b), tables)
    upper_table = reduce((a, b) -> max.(a, b), tables)
    @show typeof(lower_table)
    @show typeof(upper_table)
    return CredalPosterior(
        posteriors,
        Factor(factors[1].vars, lower_table),
        Factor(factors[1].vars, upper_table),
        posteriors[1].schema,
        query,
        evidence
    )
end

function extreme_bayesian_networks(cn::CredalNetwork)
    node_extremes = map(EnhancedBayesianNetworks._extreme_nodes, cn.nodes)
    combinations = Iterators.product(node_extremes...)
    bns = BayesianNetwork[]
    for nodes in combinations
        bn = BayesianNetwork(collect(nodes))
        bn.A = copy(cn.A)
        bn.topology = copy(cn.topology)
        push!(bns, bn)
    end
    return bns
end