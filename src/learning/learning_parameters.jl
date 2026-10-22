"""
    learn(dag::DirectAcyclicGraph, df::DataFrame, estimator::ParameterEstimator; max_iter::Int = 100, tol::Real = 1.0e-4, progress::Bool = isinteractive())
    learn(dag::DirectAcyclicGraph, df::DataFrame; max_iter::Int = 100, tol::Real = 1.0e-4, progress::Bool = isinteractive())

Learn the CPTs of `dag` from `df`. Either pass one `estimator` for every node, or declare one per node with [`add_node!`](@ref) and call the two-argument form. The two exclude each other, and a DAG declaring an estimator for only some of its nodes is rejected here rather than at [`add_node!`](@ref), since a DAG under construction is legitimately partial.

Every node of `dag` needs a `Symbol` column in `df`; other columns are ignored with a warning. A node's domain is the states observed in `df` together with any declared through [`add_node!`](@ref). A [`PreciseEstimator`](@ref) writes numbers and an [`ImpreciseEstimator`](@ref) writes `Interval`s; one interval anywhere gives a [`CredalNetwork`](@ref), otherwise a [`BayesianNetwork`](@ref).

`missing` entries are allowed, and the estimators fix how they are treated:

  - every node precise: expectation maximisation, which assumes the data are **missing at random**. `max_iter` and `tol` bound the iteration and are read on no other path.
  - every node imprecise: every entry is bounded over all completions of the data, assuming nothing about the missingness.
  - [`QuesenberryHurst`](@ref) on incomplete data, or a DAG mixing precise and imprecise nodes on incomplete data: refused with an error.

The returned network is fully specified but unordered, so call [`order!`](@ref) before inference or sampling. `progress` defaults to `isinteractive()`, silent in scripts, tests and docs.

# Examples

One estimator for the whole DAG:

```julia
bn = learn(dag, df, PreciseDirichletModel())       # maximum likelihood
bn = learn(dag, df, PreciseDirichletModel(1))      # Laplace smoothing
cn = learn(dag, df, ImpreciseDirichletModel())     # IDM with s = 1
cn = learn(dag, df, QuesenberryHurst(0.99))        # 99% simultaneous intervals
order!(cn)
```

Or one per node, where a single imprecise node makes the result credal:

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
        dag::DirectAcyclicGraph,
        df::DataFrame,
        estimator::ParameterEstimator;
        max_iter::Int = 100,
        tol::Real = 1.0e-4,
        progress::Bool = isinteractive()
    )
    declared = [n.name for n in dag.nodes if haskey(dag.estimators, n.name)]
    if !isempty(declared)
        error("Invalid call: node(s) $declared declare their own estimator; call learn(dag, df) to fit with the declared ones, or build the DAG without estimators to fit every node with a single one")
    end
    nodes = _fit_cpts(dag, df, _uniform_estimators(dag, estimator); max_iter = max_iter, tol = tol, progress = progress)
    return _dispatch(nodes, copy(dag.topology), copy(dag.A))
end

function learn(
        dag::DirectAcyclicGraph,
        df::DataFrame;
        max_iter::Int = 100,
        tol::Real = 1.0e-4,
        progress::Bool = isinteractive()
    )
    undeclared = [n.name for n in dag.nodes if !haskey(dag.estimators, n.name)]
    if length(undeclared) == length(dag.nodes)
        error("Invalid DAG: no node declares an estimator; declare one for every node with add_node!, or pass a single estimator to learn")
    elseif !isempty(undeclared)
        declared = [n.name for n in dag.nodes if haskey(dag.estimators, n.name)]
        error("Invalid DAG: node(s) $undeclared declare no estimator while $declared do; declare one for every node or none at all")
    end
    nodes = _fit_cpts(dag, df, dag.estimators; max_iter = max_iter, tol = tol, progress = progress)
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

# Which treatment incomplete data gets is fixed by the estimators rather than chosen by the caller: a precise network is completed by expectation maximisation, an imprecise one by bounding every entry over the completions of the data. This is also where the two combinations with no coherent answer are refused.
function _verify_missingness(
        dag::DirectAcyclicGraph,
        df::DataFrame,
        estimators::AbstractDict{Symbol, <:ParameterEstimator}
    )
    incomplete = [n.name for n in dag.nodes if any(ismissing, df[!, n.name])]
    if isempty(incomplete)
        return :complete
    end
    refused = sort([name for (name, e) in estimators if e isa QuesenberryHurst])
    if !isempty(refused)
        error("Invalid estimator: column(s) $incomplete contain missing values, which QuesenberryHurst on node(s) $refused cannot be fitted from; its simultaneous confidence region has no closed-form envelope over the completions of the data, and bounding each entry separately would no longer carry the stated confidence. Use ImpreciseDirichletModel, whose bounds over the completions are exact")
    end
    imprecise = sort([name for (name, e) in estimators if e isa ImpreciseEstimator])
    precise = sort([name for (name, e) in estimators if e isa PreciseEstimator])
    if !isempty(imprecise) && !isempty(precise)
        error("Invalid DAG: column(s) $incomplete contain missing values, and node(s) $precise are estimated precisely while $imprecise are not; expectation maximisation is a property of the whole network while bounding over the completions is local to a node, so the two treatments cannot be mixed. Declare one kind of estimator throughout, or complete the data")
    end
    return isempty(imprecise) ? :em : :envelope
end

# Domain of every node: the states declared on the DAG (see `add_node!`) together with those observed in `df`, sorted so that parent configurations are always enumerated in the same order. `skipmissing` keeps `missing` out of the domain. It marks an absent observation, not a state of the node.
function _domains(dag::DirectAcyclicGraph, df::DataFrame)
    domain(name) = sort(unique(vcat(get(dag.states, name, Symbol[]), collect(skipmissing(df[!, name])))))
    return Dict{Symbol, Vector{Symbol}}(n.name => domain(n.name) for n in dag.nodes)
end

# Counts of a single parent configuration. Two masks rather than one: `definite` selects the rows whose parents are all observed and all sit in the states `parent_config` prescribes, while `consistent` selects the rows that could be *completed* as this configuration, that is every observed parent matches but at least one is missing. For a root the loops never run, `definite` stays all-true (the one configuration covering every row) and `consistent` stays all-false, since with no parents nothing can be ambiguous.
function _counts(
        df::DataFrame,
        node_name::Symbol,
        node_states::Vector{Symbol},
        parent_config,
        incomplete::Bool
    )
    column = df[!, node_name]
    definite = trues(nrow(df))
    for (parent, state) in parent_config
        definite .&= isequal.(df[!, parent], state)
    end
    counts = Int[count(definite .& isequal.(column, s)) for s in node_states]
    if !incomplete
        return ConfigurationCounts(counts, sum(counts), 0, zeros(Int, length(node_states)), 0)
    end
    consistent = trues(nrow(df))
    anymissing = falses(nrow(df))
    for (parent, state) in parent_config
        col = df[!, parent]
        unknown = ismissing.(col)
        consistent .&= unknown .| isequal.(col, state)
        anymissing .|= unknown
    end
    consistent .&= anymissing
    unobserved = ismissing.(column)
    return ConfigurationCounts(
        counts,
        sum(counts),
        count(definite .& unobserved),
        Int[count(consistent .& isequal.(column, s)) for s in node_states],
        count(consistent .& unobserved),
    )
end

# EM's M-step: the E-step has already expanded each incomplete row into weighted completions, so every frequency is a sum of responsibilities and nothing is left incomplete. Dispatches apart from the `Bool` method by the type of the last argument.
function _counts(
        df::DataFrame,
        node_name::Symbol,
        node_states::Vector{Symbol},
        parent_config,
        weights::AbstractVector{<:Real}
    )
    column = df[!, node_name]
    definite = trues(nrow(df))
    for (parent, state) in parent_config
        definite .&= isequal.(df[!, parent], state)
    end
    counts = Float64[sum(weights[definite .& isequal.(column, s)]; init = 0.0) for s in node_states]
    return ConfigurationCounts(counts, sum(counts), 0.0, zeros(Float64, length(node_states)), 0.0)
end

function _fit_cpts(
        dag::DirectAcyclicGraph,
        df::DataFrame,
        estimators;
        max_iter::Int = 100,
        tol::Real = 1.0e-4,
        progress::Bool = isinteractive()
    )
    if nrow(df) == 0
        error("Invalid data: the data frame has no rows, there is nothing to learn from")
    end
    _verify_columns(dag, df)
    treatment = _verify_missingness(dag, df, estimators)
    domains = _domains(dag, df)
    if treatment === :em
        return _em(dag, df, domains, estimators, max_iter, tol, progress)
    else
        return _fit_loop(dag, df, estimators, domains; has_missing = treatment === :envelope, progress = progress)
    end
end

# The walk itself, with no verification: `_fit_cpts` has already checked the columns and chosen the treatment, and EM reaches this once per iteration on its own expanded frame, whose extra `:weight` column would otherwise trip `_verify_columns` on every pass. `has_missing` and `weights` are alternatives, not companions: the first bounds each entry over the completions of the data, the second fits from responsibilities the E-step has already resolved, so nothing is left incomplete for it to bound.
function _fit_loop(
        dag::DirectAcyclicGraph,
        df::DataFrame,
        estimators,
        domains;
        has_missing::Bool = false,
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
        progress::Bool = false
    )
    nodes = deepcopy(dag.nodes)
    meter = Progress(length(nodes); desc = "Fitting CPTs ", enabled = progress)
    for node in nodes
        node_name = node.name
        par = parents(dag, node_name)
        node_states = domains[node_name]
        for config in Iterators.product((domains[p] for p in par)...)
            parent_config = [par[i] => config[i] for i in eachindex(par)]
            if isnothing(weights)
                config_counts = _counts(df, node_name, node_states, parent_config, has_missing)
            else
                config_counts = _counts(df, node_name, node_states, parent_config, weights)
            end
            probabilities = _estimate(estimators[node_name], config_counts)
            for (i, state) in enumerate(node_states)
                node[parent_config..., node_name => state] = probabilities[i]
            end
        end
        next!(meter)
    end
    return nodes
end

# ==================================================================================================
# Expectation maximisation
#
# The precise treatment of incomplete data, reached only when every node is estimated precisely. The E-step replaces each incomplete row by its weighted completions under the current network, and the M-step is the ordinary fitting walk over those weights, so the estimator a node declares applies unchanged: `_estimate(PreciseDirichletModel(α), ...)` IS the M-step formula. Unlike bounding over the completions, this asserts the data are missing at random, which is why `learn` never reaches it with an imprecise estimator.
# ==================================================================================================

function _em(
        dag::DirectAcyclicGraph, df::DataFrame, domains, estimators, max_iter::Int, tol::Real, progress::Bool
    )
    if max_iter < 1
        error("Invalid call: max_iter must be at least 1, got $max_iter")
    end
    if tol <= 0
        error("Invalid call: tol must be strictly positive, got $(repr(tol))")
    end
    bn = _em_uniform(dag, domains)
    meter = Progress(max_iter; desc = "EM iteration ", enabled = progress)
    for _ in 1:max_iter
        completed = _em_estep(dag, df, bn, domains)
        new_bn = BayesianNetwork(
            _fit_loop(dag, completed, estimators, domains; weights = completed[!, :weight]),
            copy(dag.topology), copy(dag.A)
        )
        change = _em_maxchange(bn, new_bn)
        bn = new_bn
        next!(meter; showvalues = () -> [("max Δ", change)])
        if change < tol
            finish!(meter)
            break
        end
    end
    return bn.nodes
end

# Uniform starting network: every conditional distribution is 1/k over the node's domain.
function _em_uniform(dag::DirectAcyclicGraph, domains)
    nodes = deepcopy(dag.nodes)
    for node in nodes
        n = node.name
        par = parents(dag, n)
        node_states = domains[n]
        k = length(node_states)
        for config in Iterators.product((domains[p] for p in par)...)
            pkeys = [par[i] => config[i] for i in eachindex(par)]
            for s in node_states
                node[pkeys..., n => s] = 1 / k
            end
        end
    end
    return BayesianNetwork(nodes, copy(dag.topology), copy(dag.A))
end

# E-step: expand every row into weighted completions of its missing variables. A complete row keeps
# weight 1; an incomplete row yields one completed row per assignment of its missing variables,
# weighted by P(missing | observed) = normalized joint_probability under bn. Returns the completed
# data with an added :weight column.
function _em_estep(dag::DirectAcyclicGraph, df::DataFrame, bn::BayesianNetwork, domains)
    node_names = [n.name for n in dag.nodes]
    columns = Dict(c => Symbol[] for c in node_names)
    weights = Float64[]
    for row in eachrow(df)
        missing_vars = [c for c in node_names if ismissing(row[c])]
        if isempty(missing_vars)
            for c in node_names
                push!(columns[c], row[c])
            end
            push!(weights, 1.0)
            continue
        end
        observed = [c => row[c] for c in node_names if !ismissing(row[c])]
        completions = collect(Iterators.product((domains[v] for v in missing_vars)...))
        w = map(completions) do combo
            scenario = Evidence(vcat(observed, [missing_vars[j] => combo[j] for j in eachindex(missing_vars)]))
            joint_probability(bn, scenario)
        end
        total = sum(w)
        w = total == 0 ? fill(1 / length(w), length(w)) : w ./ total
        for (combo, weight) in zip(completions, w)
            assignment = Dict(observed)
            for j in eachindex(missing_vars)
                assignment[missing_vars[j]] = combo[j]
            end
            for c in node_names
                push!(columns[c], assignment[c])
            end
            push!(weights, weight)
        end
    end
    completed = DataFrame(columns)
    completed[!, :weight] = weights
    return completed
end

# Largest change in any CPT entry between successive iterations (rows are built in the same order).
_em_maxchange(a::BayesianNetwork, b::BayesianNetwork) = maximum(maximum(abs.(x.cpt.data.Π .- y.cpt.data.Π)) for (x, y) in zip(a.nodes, b.nodes))

# One estimator for every node, so that `_fit_cpts` only ever sees the per-node form.
_uniform_estimators(dag::DirectAcyclicGraph, estimator::ParameterEstimator) =
    Dict{Symbol, ParameterEstimator}(n.name => estimator for n in dag.nodes)
