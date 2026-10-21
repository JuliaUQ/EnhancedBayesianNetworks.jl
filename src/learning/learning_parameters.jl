"""
    learn(dag::DirectAcyclicGraph, df::DataFrame, estimator::ParameterEstimator; progress::Bool = isinteractive())

Learn the CPTs of `dag` from `df` with `estimator`, which decides both how each conditional distribution is estimated and what kind of network comes back:

  - a [`PreciseEstimator`](@ref), namely [`PreciseDirichletModel`](@ref), gives a [`BayesianNetwork`](@ref).
  - an [`ImpreciseEstimator`](@ref), namely [`ImpreciseDirichletModel`](@ref) or [`QuesenberryHurst`](@ref), gives a [`CredalNetwork`](@ref).

Every node of `dag` must have a complete `Symbol` column in `df`; columns of `df` that are not nodes of `dag` are ignored with a warning.

A node's domain is the set of states observed in `df` together with any declared through [`add_node!`](@ref), so a state that never occurs still receives a CPT entry, and a parent configuration that never occurs is estimated from `total == 0`: uniform under maximum likelihood, vacuous under either imprecise estimator.

The returned network is fully specified but unordered, so call [`order!`](@ref) before inference or sampling. `progress` shows a bar over the fitted nodes and defaults to `isinteractive()`, visible in the REPL and silent in scripts, tests and docs.

# Examples
```julia
bn = learn(dag, df, PreciseDirichletModel())       # maximum likelihood
bn = learn(dag, df, PreciseDirichletModel(1))      # Laplace smoothing
cn = learn(dag, df, ImpreciseDirichletModel())     # IDM with s = 1
cn = learn(dag, df, QuesenberryHurst(0.99))        # 99% simultaneous intervals
order!(cn)
```
"""
function learn(
        dag::DirectAcyclicGraph, df::DataFrame, estimator::PreciseEstimator; progress::Bool = isinteractive()
    )
    nodes = _fit_cpts(dag, df, estimator; progress = progress)
    return BayesianNetwork(nodes, copy(dag.topology), copy(dag.A))
end

function learn(
        dag::DirectAcyclicGraph, df::DataFrame, estimator::ImpreciseEstimator; progress::Bool = isinteractive()
    )
    nodes = _fit_cpts(dag, df, estimator; progress = progress)
    return CredalNetwork(nodes, copy(dag.topology), copy(dag.A))
end

# ==================================================================================================
# Shared fitting engine
#
# One walk over the DAG, shared by every estimator: resolve each node's domain, enumerate the parent configurations, count the rows matching each, and hand the counts to `_estimate`. The estimator decides what goes into the CPT; nothing here knows whether that is a number or an `Interval`.
# ==================================================================================================

# Every node of the DAG must have a column in `df`, that is the contract between the structure and the data. The reverse direction is only a warning: a wide table may legitimately carry identifiers, timestamps or measurements taken for another purpose, and no estimator ever reads them.
function _verify_columns(dag::DirectAcyclicGraph, df::DataFrame)
    node_names = [n.name for n in dag.nodes]
    extra = setdiff(propertynames(df), node_names)
    if !isempty(extra)
        @warn("Data contains column(s) $extra that are not nodes of the DAG; they are ignored in the parameter estimation")
    end
    absent = setdiff(node_names, propertynames(df))
    if !isempty(absent)
        error("Invalid data: node(s) $absent of the DAG have no column in the data")
    end
    wrongtype = [n for n in node_names if !(eltype(df[!, n]) <: Union{Missing, Symbol})]
    if !isempty(wrongtype)
        types = join(["$n::$(eltype(df[!, n]))" for n in wrongtype], ", ")
        error("Invalid data: node states are Symbols, but column(s) $types are not")
    end
    return nothing
end

# Check that there are no `missing` values in the DataFrame
function _verify_complete(dag::DirectAcyclicGraph, df::DataFrame)
    incomplete = [n.name for n in dag.nodes if any(ismissing, df[!, n.name])]
    if !isempty(incomplete)
        error("Invalid data: column(s) $incomplete contain missing values")
    end
    return nothing
end

# Domain of every node: the states declared on the DAG (see `add_node!`) together with those observed in `df`, sorted so that parent configurations are always enumerated in the same order. `skipmissing` keeps `missing` out of the domain. It marks an absent observation, not a state of the node.
function _domains(dag::DirectAcyclicGraph, df::DataFrame)
    domain(name) = sort(unique(vcat(get(dag.states, name, Symbol[]), collect(skipmissing(df[!, name])))))
    return Dict{Symbol, Vector{Symbol}}(n.name => domain(n.name) for n in dag.nodes)
end

# Counts of a single parent configuration. `mask` selects the rows whose parents all sit in the states `parent_config` prescribes; for a root it is empty, the loop body never runs, and the mask stays all-true, which is the one configuration covering every row. `total` is read off the mask andit is the normaliser the estimators divide by.
function _counts(df::DataFrame, node_name::Symbol, node_states::Vector{Symbol}, parent_config)
    mask = trues(nrow(df))
    for (parent, state) in parent_config
        mask .&= isequal.(df[!, parent], state)
    end
    column = df[!, node_name]
    counts = [count(mask .& isequal.(column, s)) for s in node_states]
    return counts, count(mask)
end

# One walk over the DAG: resolve the domains, enumerate each node's parent configurations, count the rows matching each and hand the counts to `_estimate`. Returns the fitted nodes rather than a finished network, because the same loop serves both kinds: `learn` wraps them in a `BayesianNetwork` or a `CredalNetwork` according to the estimator it was given. Nothing here knows whether a CPT entry is a number or an `Interval`. `parent_config` is built once per configuration and used twice, as the mask recipe for `_counts` and as the CPT key prefix. For a root it is empty, which is one configuration covering every row.
# states invariancy: `_estimate` returns a vector of probability coherent with the vector coming from `_counts`, which is applied only once in the `_fit_cpts` and only over `domains[node_name]`. Therefore the states order is garantee
function _fit_cpts(
        dag::DirectAcyclicGraph, df::DataFrame, estimator::ParameterEstimator; progress::Bool = isinteractive()
    )
    if nrow(df) == 0
        error("Invalid data: the data frame has no rows, there is nothing to learn from")
    end
    _verify_columns(dag, df)
    _verify_complete(dag, df)
    domains = _domains(dag, df)
    nodes = deepcopy(dag.nodes)
    meter = Progress(length(nodes); desc = "Fitting CPTs ", enabled = progress)
    for node in nodes
        node_name = node.name
        par = parents(dag, node_name)
        node_states = domains[node_name]
        for config in Iterators.product((domains[p] for p in par)...)
            parent_config = [par[i] => config[i] for i in eachindex(par)]
            counts, total = _counts(df, node_name, node_states, parent_config)
            probabilities = _estimate(estimator, counts, total)
            for (i, state) in enumerate(node_states)
                node[parent_config..., node_name => state] = probabilities[i]
            end
        end
        next!(meter)
    end
    return nodes
end
