"""
    learn(dag::DirectAcyclicGraph, df::DataFrame, estimator::ParameterEstimator; progress::Bool = isinteractive())
    learn(dag::DirectAcyclicGraph, df::DataFrame; progress::Bool = isinteractive())

Learn the CPTs of `dag` from `df`. The estimator can be supplied in either of two ways, and the two forms exclude each other:

  - pass a single `estimator`, which fits every node the same way. The DAG must declare none of its own.
  - declare one per node through [`add_node!`](@ref) and call `learn(dag, df)`, which lets a single DAG mix precisely and imprecisely estimated nodes. It is all or nothing: a DAG where some nodes declare an estimator and others do not is rejected here rather than at [`add_node!`](@ref), because a DAG under construction is legitimately partial and the rule only has to hold at the moment it is fitted.

Which kind of network comes back follows from the fitted CPTs rather than from the estimator: a [`PreciseEstimator`](@ref), namely [`PreciseDirichletModel`](@ref), writes numbers, while an [`ImpreciseEstimator`](@ref), namely [`ImpreciseDirichletModel`](@ref) or [`QuesenberryHurst`](@ref), writes intervals. One interval anywhere gives a [`CredalNetwork`](@ref), and otherwise the result is a [`BayesianNetwork`](@ref). The two coincide except for a node with a single state, whose probability is `1` under every estimator, so a DAG of such nodes stays precise however it was fitted.

Every node of `dag` must have a complete `Symbol` column in `df`; columns of `df` that are not nodes of `dag` are ignored with a warning.

A node's domain is the set of states observed in `df` together with any declared through [`add_node!`](@ref), so a state that never occurs still receives a CPT entry, and a parent configuration that never occurs is estimated from `total == 0`: uniform under maximum likelihood, vacuous under either imprecise estimator.

The returned network is fully specified but unordered, so call [`order!`](@ref) before inference or sampling. `progress` shows a bar over the fitted nodes and defaults to `isinteractive()`, visible in the REPL and silent in scripts, tests and docs.

# Examples

One estimator for the whole DAG:

```julia
bn = learn(dag, df, PreciseDirichletModel())       # maximum likelihood
bn = learn(dag, df, PreciseDirichletModel(1))      # Laplace smoothing
cn = learn(dag, df, ImpreciseDirichletModel())     # IDM with s = 1
cn = learn(dag, df, QuesenberryHurst(0.99))        # 99% simultaneous intervals
order!(cn)
```

Or one per node, declared on the DAG, where a single imprecise node is enough to make the result credal:

```julia
dag = DirectAcyclicGraph()
add_node!(dag, :W; estimator = PreciseDirichletModel())
add_node!(dag, :R; parents = [:W], estimator = ImpreciseDirichletModel())
add_node!(dag, :G; parents = [:R], estimator = PreciseDirichletModel())

cn = learn(dag, df)                                # -> CredalNetwork, only :R is imprecise
order!(cn)
```
"""
function learn(
        dag::DirectAcyclicGraph, df::DataFrame, estimator::ParameterEstimator; progress::Bool = isinteractive()
    )
    declared = [n.name for n in dag.nodes if haskey(dag.estimators, n.name)]
    if !isempty(declared)
        error("Invalid call: node(s) $declared declare their own estimator; call learn(dag, df) to fit with the declared ones, or build the DAG without estimators to fit every node with a single one")
    end
    nodes = _fit_cpts(dag, df, _uniform_estimators(dag, estimator); progress = progress)
    return _dispatch(nodes, copy(dag.topology), copy(dag.A))
end

function learn(
        dag::DirectAcyclicGraph, df::DataFrame; progress::Bool = isinteractive()
    )
    undeclared = [n.name for n in dag.nodes if !haskey(dag.estimators, n.name)]
    if length(undeclared) == length(dag.nodes)
        error("Invalid DAG: no node declares an estimator; declare one for every node with add_node!, or pass a single estimator to learn")
    elseif !isempty(undeclared)
        declared = [n.name for n in dag.nodes if haskey(dag.estimators, n.name)]
        error("Invalid DAG: node(s) $undeclared declare no estimator while $declared do; declare one for every node or none at all")
    end
    nodes = _fit_cpts(dag, df, dag.estimators; progress = progress)
    return _dispatch(nodes, copy(dag.topology), copy(dag.A))
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
        dag::DirectAcyclicGraph, df::DataFrame, estimators::AbstractDict{Symbol, <:ParameterEstimator}; progress::Bool = isinteractive()
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
            probabilities = _estimate(estimators[node_name], counts, total)
            for (i, state) in enumerate(node_states)
                node[parent_config..., node_name => state] = probabilities[i]
            end
        end
        next!(meter)
    end
    return nodes
end

# One estimator for every node, so that `_fit_cpts` only ever sees the per-node form.
_uniform_estimators(dag::DirectAcyclicGraph, estimator::ParameterEstimator) =
    Dict{Symbol, ParameterEstimator}(n.name => estimator for n in dag.nodes)
