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
S[:W => :sunny, :S => :on] = 0.9; S[:W => :sunny, :S => :off] = 0.1
S[:W => :cloudy, :S => :on] = 0.2; S[:W => :cloudy, :S => :off] = 0.8
bn = BayesianNetwork([W, S]); add_child!(bn, :W, :S); order!(bn)

p = infer(bn, :S, Evidence(:W => :sunny))   # Posterior P(S | W=sunny)
```
"""
struct Posterior{T, A <: AbstractArray{T}}
    factor::Factor{T, A}
    schema::NetworkSchema
    query::Vector{Symbol}
    evidence::Evidence
end


"""
    CredalPosterior

The result of infer on a CredalNetwork: lower and upper posterior probabilities
over the query given evidence. lower/upper are the element-wise min/max Factors over the
posteriors obtained from the extreme Bayesian networks of the credal set that admit the evidence.
An extreme network under which the evidence has probability zero gives a 0/0 posterior and says
nothing about the conditional, so it is left out of the bounds rather than folded into them; this
is regular extension, and discarded counts the extremes dropped that way. schema, query, and
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
struct CredalPosterior{T, A <: AbstractArray{T}}
    posteriors::Vector{<:Posterior}
    lower::Factor{T, A}
    upper::Factor{T, A}
    schema::NetworkSchema
    query::Vector{Symbol}
    evidence::Evidence
    discarded::Int
end

CredalPosterior(posteriors, lower, upper, schema, query, evidence) =
    CredalPosterior(posteriors, lower, upper, schema, query, evidence, 0)

"""
    infer(bn::BayesianNetwork, query, evidence::Evidence, scorefun = fill_factor_score; progress::Bool = isinteractive())
    infer(cn::CredalNetwork, query, evidence::Evidence, scorefun = fill_factor_score; progress::Bool = isinteractive(), tol::Real = 0.0)

Compute the posterior over query (a Symbol or a vector of them) given evidence, by variable
elimination. Returns a Posterior for a Bayesian network, or a CredalPosterior with
lower/upper bounds over the credal set's extreme networks for a credal one. scorefun selects the
elimination-ordering heuristic — fill_factor_score (default), fill_score, or
factor_score. The query must not overlap the evidence, and both must name existing nodes/states.
`progress` shows a progress bar over the work — the eliminated variables for a Bayesian network,
the extreme networks for a credal one — and defaults to `isinteractive()` (shown in the REPL,
silent in scripts, tests, and docs); force it with `progress=true` / `progress=false`.

On a credal network the bounds are taken over the extreme networks under which the evidence is
possible; one under which P(evidence) is zero gives a 0/0 posterior and is discarded (regular
extension). `tol` is the threshold at or below which P(evidence) counts as impossible, and
defaults to 0.0, which discards exactly the structurally degenerate extremes. If the evidence is
impossible under every extreme network, so that its upper probability is zero, infer raises an
error rather than returning a vacuous interval.

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
        query::Union{Symbol, Vector{Symbol}},
        evidence::Evidence,
        scorefun = fill_factor_score;
        progress::Bool = isinteractive()
    )
    query = _wrap(query)
    _verify_query(query, bn, evidence)
    _verify_evidence(evidence, bn)

    posterior, _ = _infer_ve(bn, query, evidence, scorefun; progress = progress)
    return posterior
end

# One variable-elimination pass over a precise network: the posterior, and P(evidence) under it.
# `infer` keeps only the posterior; credal inference needs P(evidence) to tell whether a given
# extreme network admits the evidence at all.
function _infer_ve(
        bn::BayesianNetwork,
        query::Vector{Symbol},
        evidence::Evidence,
        scorefun;
        progress::Bool = false
    )
    ns = NetworkSchema(bn)
    ig = InteractionGraph(bn)
    factors = _factorize(bn)
    query_vars = _query_to_idx(query, ns)
    evidence_idx = _evidence_to_idx(evidence, ns)
    order = _sort_nodes(ig, ns, scorefun)
    result, evidence_probability = _ve(factors, order, query_vars, evidence_idx; progress = progress)
    return Posterior(result, ns, query, evidence), evidence_probability
end

function infer(
        cn::CredalNetwork,
        query::Union{Symbol, Vector{Symbol}},
        evidence::Evidence,
        scorefun = fill_factor_score;
        progress::Bool = isinteractive(),
        tol::Real = 0.0
    )
    query = _wrap(query)
    _verify_query(query, cn, evidence)
    _verify_evidence(evidence, cn)

    posteriors = Posterior[]
    discarded = 0
    bns = _extreme_bayesian_networks(cn)
    p = Progress(length(bns); desc = "Inferring over $(length(bns)) BNs ", enabled = progress)
    for bn in bns
        posterior, evidence_probability = _infer_ve(bn, query, evidence, scorefun; progress = false)
        # Regular extension: an extreme network under which the evidence is impossible says nothing
        # about the conditional — its posterior is 0/0 — so it is dropped rather than folded into the
        # bounds, where a single NaN would swallow both of them.
        if evidence_probability > tol
            push!(posteriors, posterior)
        else
            discarded += 1
        end
        next!(p)
    end
    if isempty(posteriors)
        evidence_str = "[" * join(["$(repr(k)) => $(repr(v))" for (k, v) in evidence], ", ") * "]"
        error("Invalid Evidence: evidence $evidence_str has upper probability zero, it is impossible under every measure of the credal set, therefore the conditional probability is undefined")
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
        evidence,
        discarded
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
