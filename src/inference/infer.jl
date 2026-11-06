"""
    Posterior

The result of [`infer`](@ref) on a [`BayesianNetwork`](@ref): the posterior distribution over the `query` variables given `evidence`. Holds the resulting probability `Factor`, the `NetworkSchema` needed to map ids back to names/states, and the original `query`/`evidence`. Display it to see the labelled probability table.

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

The result of infer on a CredalNetwork: lower and upper posterior probabilities over the query given evidence. lower/upper are the element-wise min/max Factors over the posteriors obtained from the extreme Bayesian networks of the credal set that admit the evidence. An extreme network under which the evidence has probability zero gives a 0/0 posterior and says nothing about the conditional, so it is left out of the bounds rather than folded into them; this is regular extension, and discarded counts the extremes dropped that way. schema, query, and evidence mirror Posterior. Display it to see the labelled [lower, upper] table.

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

# Six-argument constructor kept for compatibility: a CredalPosterior built without a discarded
# count is one where no extreme network was discarded.
function CredalPosterior(
        posteriors::Vector{<:Posterior},
        lower::Factor{T, A},
        upper::Factor{T, A},
        schema::NetworkSchema,
        query::Vector{Symbol},
        evidence::Evidence
    ) where {T, A <: AbstractArray{T}}
    return CredalPosterior(posteriors, lower, upper, schema, query, evidence, 0)
end

"""
    infer(bn::BayesianNetwork, query, evidence::Evidence, scorefun = fill_factor_score; progress::Bool = isinteractive())
    infer(cn::CredalNetwork, query, evidence::Evidence, scorefun = fill_factor_score; progress::Bool = isinteractive(), tol::Real = 0.0, max_networks::Integer = 10^6)

Compute the posterior over query (a Symbol or a vector of them) given evidence, by variable elimination. Returns a Posterior for a Bayesian network, or a CredalPosterior with lower/upper bounds over the credal set's extreme networks for a credal one. scorefun selects the elimination-ordering heuristic — fill_factor_score (default), fill_score, or factor_score. The query must not overlap the evidence, and both must name existing nodes/states. `progress` shows a progress bar over the work — the eliminated variables for a Bayesian network, the extreme networks for a credal one — and defaults to `isinteractive()` (shown in the REPL, silent in scripts, tests, and docs); force it with `progress=true` / `progress=false`.

On a credal network the bounds are taken over the extreme networks under which the evidence is possible; one under which P(evidence) is zero gives a 0/0 posterior and is discarded (regular extension). `tol` is the threshold at or below which P(evidence) counts as impossible, and defaults to 0.0, which discards exactly the structurally degenerate extremes. If the evidence is impossible under every extreme network, so that its upper probability is zero, infer raises an error rather than returning a vacuous interval.

Exact credal inference runs one variable elimination per extreme network, and their number is the product of the vertex counts of every conditional distribution, so it grows exponentially with the imprecise nodes. `infer` counts them before building any and raises an error when the count exceeds `max_networks` (default `10^6`); raise it to accept a longer run. The extreme networks are inferred in parallel over the threads Julia was started with, so start it with `julia --threads=auto` to use every core; the result does not depend on the number of threads.

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

# One variable-elimination pass over a precise network: the posterior, and P(evidence) under it. `infer` keeps only the posterior.
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
        tol::Real = 0.0,
        max_networks::Integer = 10^6
    )
    if !isfinite(tol) || tol < 0
        error("Invalid tol: tol must be finite and nonnegative, got $(repr(tol))")
    end
    query = _wrap(query)
    _verify_query(query, cn, evidence)
    _verify_evidence(evidence, cn)

    # Count before building: enumerating the extreme networks is what costs, so refuse an infeasible network up front rather than hanging in the enumeration.
    n = _n_extreme_bayesian_networks(cn)
    if n > max_networks
        error("Invalid CredalNetwork: exact credal inference enumerates every extreme Bayesian network, and this one has $(n <= 10^12 ? n : "≈ 10^$(round(Float64(log10(n)); digits = 1))") of them, above max_networks = $max_networks; raise max_networks or reduce the imprecision of the network")
    end
    # Every extreme network has the structure of `cn` and only its CPT entries differ, so the schema, the elimination order and the factor of each extreme node are computed once, and an extreme network is just a choice of one factor per node.
    node_extremes = map(_extreme_nodes, cn.nodes)
    sizes = length.(node_extremes)
    template = BayesianNetwork(_extreme_combination(node_extremes, sizes, 0), copy(cn.topology), copy(cn.A))
    ns = NetworkSchema(template)
    order = _sort_nodes(InteractionGraph(template), ns, scorefun)
    node_factors = [Factor[_factorize(node, ns) for node in extremes] for extremes in node_extremes]
    query_vars = _query_to_idx(query, ns)
    evidence_idx = _evidence_to_idx(evidence, ns)

    p = Progress(Int(n); desc = "Inferring over $n BNs ", enabled = progress)
    # The extreme networks are split among the available threads, a few chunks per thread so that a thread finishing early picks up more work. The chunks are joined back in order, so `posteriors` comes out the same as in a serial run.
    chunks = Iterators.partition(0:(Int(n) - 1), cld(Int(n), 4 * Threads.nthreads()))
    tasks = map(chunks) do chunk
        Threads.@spawn begin
            chunk_posteriors = Posterior[]
            chunk_discarded = 0
            for i in chunk
                extreme_factors = _extreme_combination(node_factors, sizes, i)
                result, evidence_probability = _ve(extreme_factors, order, query_vars, evidence_idx)
                # Regular extension: an extreme network under which the evidence is impossible says nothing about the conditional — its posterior is 0/0 — so it is dropped rather than folded into the bounds, where a single NaN would swallow both of them.
                if evidence_probability > tol
                    push!(chunk_posteriors, Posterior(result, ns, query, evidence))
                else
                    chunk_discarded += 1
                end
                next!(p)
            end
            (chunk_posteriors, chunk_discarded)
        end
    end
    results = fetch.(tasks)
    posteriors = reduce(vcat, first.(results))
    discarded = sum(last.(results))
    if isempty(posteriors)
        evidence_str = "[" * join(["$(repr(k)) => $(repr(v))" for (k, v) in evidence], ", ") * "]"
        # With the default tol every surviving extreme was genuinely impossible, so the conditional does not exist. With a raised tol the extremes may well have admitted the evidence and been cut by the threshold instead, which is the caller's doing and not a property of the model.
        if iszero(tol)
            error("Invalid Evidence: evidence $evidence_str has upper probability zero, it is impossible under every measure of the credal set, therefore the conditional probability is undefined")
        else
            error("Invalid Evidence: evidence $evidence_str has P(evidence) <= $(repr(tol)) under every extreme network of the credal set, so tol discarded all of them; lower tol to condition on it")
        end
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

# Number of extreme networks of `cn`, computed without building any: the product of the per-node counts.
_n_extreme_bayesian_networks(cn::CredalNetwork) = prod(_n_extreme_nodes, cn.nodes; init = big(1))

# The `i`-th combination of one element per node, in the same order as `Iterators.product`: the first node varies fastest. `node_extremes` holds per node either its extreme nodes or their factors, and the result has the same element type. The combinations are walked with an integer counter rather than `Iterators.product(node_extremes...)`, whose splat makes the compiler specialise on a tuple as long as the network and costs tens of seconds on the first call for a network of a hundred nodes; it also lets each thread start at its own index.
function _extreme_combination(node_extremes, sizes, i::Integer)
    picked = Vector{eltype(eltype(node_extremes))}(undef, length(sizes))
    for k in eachindex(sizes)
        i, r = divrem(i, sizes[k])
        picked[k] = node_extremes[k][r + 1]
    end
    return picked
end
