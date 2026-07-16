
"""
    Posterior

The result of [`infer`](@ref) on a [`BayesianNetwork`](@ref): the posterior distribution over the
`query` variables given `evidence`. Holds the resulting probability `Factor`, the `NetworkSchema`
needed to map ids back to names/states, and the original `query`/`evidence`. Display it to see the
labelled probability table.

# Examples
```julia
W = DiscreteNode(:W); W[:W => :sunny] = 0.5; W[:W => :cloudy] = 0.5
S = DiscreteNode(:S, [:W])
S[:W => :sunny,  :S => :on] = 0.9; S[:W => :sunny,  :S => :off] = 0.1
S[:W => :cloudy, :S => :on] = 0.2; S[:W => :cloudy, :S => :off] = 0.8
bn = BayesianNetwork([W, S]); add_child!(bn, :W, :S); order!(bn)

p = infer(bn, :S, Evidence(:W => :sunny))   # Posterior P(S | W=sunny)
```
"""
struct Posterior{T,A<:AbstractArray{T}}
    factor::Factor{T,A}
    schema::NetworkSchema
    query::Vector{Symbol}
    evidence::Evidence
end


"""
    CredalPosterior

The result of infer on a CredalNetwork: lower and upper posterior probabilities
over the query given evidence. lower/upper are the element-wise min/max Factors over the
posteriors obtained from every extreme Bayesian network of the credal set; schema, query, and
evidence mirror Posterior. Display it to see the labelled [lower, upper] table.

Examples

W = DiscreteNode(:W); W[:W => :sunny] = 0.5; W[:W => :cloudy] = 0.5
S = DiscreteNode(:S, [:W])
S[:W => :sunny,  :S => :on]  = Interval(0.8, 0.95); S[:W => :sunny,  :S => :off] = Interval(0.05, 0.2)
S[:W => :cloudy, :S => :on]  = 0.2;                 S[:W => :cloudy, :S => :off] = 0.8
cn = CredalNetwork([W, S]); add_child!(cn, :W, :S); order!(cn)

p = infer(cn, [:S], Evidence(:W => :sunny))  # CredalPosterior with lower/upper bounds
```
"""
struct CredalPosterior{T,A<:AbstractArray{T}}
    posteriors::Vector{<:Posterior}
    lower::Factor{T,A}
    upper::Factor{T,A}
    schema::NetworkSchema
    query::Vector{Symbol}
    evidence::Evidence
end

"""
    infer(bn::BayesianNetwork, query, evidence::Evidence, scorefun=fill_factor_score)
    infer(cn::CredalNetwork, query, evidence::Evidence, scorefun=fill_factor_score)

Compute the posterior over query (a Symbol or a vector of them) given evidence, by variable
elimination. Returns a Posterior for a Bayesian network, or a CredalPosterior with
lower/upper bounds over the credal set's extreme networks for a credal one. scorefun selects the
elimination-ordering heuristic — fill_factor_score (default), fill_score, or
factor_score. The query must not overlap the evidence, and both must name existing nodes/states.

Examples

W = DiscreteNode(:W); W[:W => :sunny] = 0.5; W[:W => :cloudy] = 0.5
S = DiscreteNode(:S, [:W])
S[:W => :sunny,  :S => :on] = 0.9; S[:W => :sunny,  :S => :off] = 0.1
S[:W => :cloudy, :S => :on] = 0.2; S[:W => :cloudy, :S => :off] = 0.8
bn = BayesianNetwork([W, S]); add_child!(bn, :W, :S); order!(bn)

infer(bn, :S, Evidence(:W => :sunny))       # Posterior P(S | W=sunny)
infer(bn, :S, Evidence())                   # prior marginal P(S)
```
"""
function infer(
    bn::BayesianNetwork,
    query::Union{Symbol,Vector{Symbol}},
    evidence::Evidence,
    scorefun=fill_factor_score
)
    query = _wrap(query)
    _verify_query(query, bn, evidence)
    _verify_evidence(evidence, bn)

    ns = NetworkSchema(bn)
    ig = InteractionGraph(bn)
    factors = _factorize(bn)
    query_vars = query_to_idx(query, ns)
    evidence_idx = evidence_to_idx(evidence, ns)
    order = sort_nodes(ig, ns, scorefun)
    result = _ve(factors, order, query_vars, evidence_idx)
    return Posterior(result, ns, query, evidence)
end

function infer(
    cn::CredalNetwork,
    query::Union{Symbol,Vector{Symbol}},
    evidence::Evidence,
    scorefun=fill_factor_score
)
    query = _wrap(query)
    _verify_query(query, cn, evidence)
    _verify_evidence(evidence, cn)

    posteriors = Posterior[]
    bns = _extreme_bayesian_networks(cn)
    @info("Performing inference over $(length(bns)) BNs")
    for bn in bns
        push!(posteriors, infer(bn, query, evidence, scorefun))
    end
    factors = getproperty.(posteriors, :factor)
    tables = getproperty.(factors, :table)

    lower_table = reduce((a, b) -> min.(a, b), tables)
    upper_table = reduce((a, b) -> max.(a, b), tables)

    return CredalPosterior(
        posteriors,
        Factor(factors[1].vars, lower_table),
        Factor(factors[1].vars, upper_table),
        posteriors[1].schema,
        query,
        evidence
    )
end

function _extreme_bayesian_networks(cn::CredalNetwork)
    node_extremes = map(EnhancedBayesianNetworks._extreme_nodes, cn.nodes)
    combinations = Iterators.product(node_extremes...)
    bns = BayesianNetwork[]
    for nodes in combinations
        bn = BayesianNetwork(collect(nodes), copy(cn.topology), copy(cn.A))
        push!(bns, bn)
    end
    return bns
end