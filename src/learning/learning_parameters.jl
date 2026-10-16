"""
    learn(dag::DirectAcyclicGraph, df::DataFrame; alpha = 0, max_iter = 100, tol = 1.0e-4, progress::Bool = isinteractive())

Learn the CPTs of `dag` from `df`, choosing the algorithm from the data: with no missing entries it
uses [`learn_parameters_mle`](@ref) (closed-form, exact); with any missing entries it uses
[`learn_parameters_em`](@ref). `alpha` is the Laplace/Dirichlet pseudo-count (both algorithms);
`max_iter` and `tol` control EM's convergence. Returns a fully-specified [`BayesianNetwork`](@ref);
call [`order!`](@ref) before inference or sampling. To force a specific algorithm, call
[`learn_parameters_mle`](@ref) or [`learn_parameters_em`](@ref) directly.
`progress` shows a progress bar — over the fitted nodes for MLE, over the EM iterations for EM — and
defaults to `isinteractive()` (shown in the REPL, silent in scripts, tests, and docs); force it with
`progress=true` / `progress=false`.

# Examples
```julia
learn(dag, df)                              # complete data -> MLE
learn(dag, df_with_missing)                 # has missing   -> EM
learn(dag, df; alpha = 1)                   # smoothing (either algorithm)
learn(dag, df_with_missing; tol = 1.0e-6, max_iter = 500)
```
"""
function learn(
        dag::DirectAcyclicGraph, df::DataFrame; alpha::Real = 0, max_iter::Int = 100, tol::Real = 1.0e-4, progress::Bool = isinteractive()
    )
    incomplete = any(n -> any(ismissing, df[!, n.name]), dag.nodes)
    if incomplete
        return learn_parameters_em(dag, df; alpha = alpha, max_iter = max_iter, tol = tol, progress = progress)
    else
        return learn_parameters_mle(dag, df; alpha = alpha, progress = progress)
    end
end

function learn_cn(dag::DirectAcyclicGraph, df::DataFrame; α::Real = 0.05, progress::Bool = isinteractive())
    statespace(col) = sort(unique(vcat(get(dag.states, col, Symbol[]), df[!, col])))
    nodes = deepcopy(dag.nodes)
    p = Progress(length(nodes); desc = "Fitting CPTs ", enabled = progress)

    for node in nodes
        n = node.name
        par = parents(dag, n)
        node_states = statespace(n)
        k = length(node_states)
        parent_states = [statespace(p) for p in par]
        for config in Iterators.product(parent_states...)
            pkeys = [par[i] => config[i] for i in eachindex(par)]
            mask = trues(nrow(df))
            for (p, c) in zip(par, config)
                mask .&= df[!, p] .== c
            end
            total = count(mask)
            for s in node_states
                if k == 1
                    node[pkeys..., n => s] = Interval(1.0, 1.0)
                else
                    cnt = count(mask .& (df[!, n] .== s))
                    A = quantile(Chisq(k - 1), 1 - α)
                    if total == 0
                        # engineering assumption, since not covered by QH
                        lower = 0.0
                        upper = 10.0^(-round(Int, log10(size(df, 1))))
                    else
                        lower = cnt == 0 ? 0.0 : (A + 2 * cnt - sqrt(A * (A + 4 * cnt * (total - cnt) / total))) / (2 * (total + A))
                        upper = cnt == total ? 1.0 : (A + 2 * cnt + sqrt(A * (A + 4 * cnt * (total - cnt) / total))) / (2 * (total + A))
                    end
                    node[pkeys..., n => s] = Interval(lower, upper)
                end
            end
        end
        next!(p)
    end
    return CredalNetwork(nodes, copy(dag.topology), copy(dag.A))
end
"""
    learn_parameters_mle(dag::DirectAcyclicGraph, df::DataFrame; alpha = 0, progress::Bool = isinteractive())

Estimate the CPTs of `dag` from complete data `df` by maximum likelihood, returning a fully-specified
[`BayesianNetwork`](@ref) (call [`order!`](@ref) on it before inference or sampling).

Each node's domain is the states observed in `df` together with any extra states declared on the
`dag`, so declared-but-unobserved states appear with probability `0` (or `alpha`-smoothed mass). For
every node and every parent configuration, `P(node = s | parents = config)` is
`(count + alpha) / (total + alpha * k)`, with `alpha` a Laplace/Dirichlet pseudo-count (`alpha = 0`
is pure MLE, `k` the number of node states). A parent configuration absent from the data falls back
to a uniform distribution. `dag` is left untouched. `progress` shows a progress bar over the nodes as
their CPTs are fitted, defaulting to `isinteractive()` (REPL only); force it with `progress=true` /
`progress=false`.

# Examples
```julia
dag = DirectAcyclicGraph()
add_node!(dag, :V, [:maybe])
add_node!(dag, :T; parents = [:V])

learned = learn_parameters_mle(dag, df)
order!(learned)
```
"""
function learn_parameters_mle(
        dag::DirectAcyclicGraph, df::DataFrame; alpha::Real = 0, progress::Bool = isinteractive()
    )
    # domain of a node = states seen in the data ∪ extra states declared on the DAG
    statespace(col) = sort(unique(vcat(get(dag.states, col, Symbol[]), df[!, col])))
    nodes = deepcopy(dag.nodes)
    p = Progress(length(nodes); desc = "Fitting CPTs ", enabled = progress)
    for node in nodes
        n = node.name
        par = parents(dag, n)
        node_states = statespace(n)
        k = length(node_states)
        parent_states = [statespace(p) for p in par]
        for config in Iterators.product(parent_states...)
            pkeys = [par[i] => config[i] for i in eachindex(par)]
            # rows whose parents match this configuration
            mask = trues(nrow(df))
            for (p, c) in zip(par, config)
                mask .&= df[!, p] .== c
            end
            total = count(mask)
            for s in node_states
                cnt = count(mask .& (df[!, n] .== s))
                prob = total == 0 ? 1 / k : (cnt + alpha) / (total + alpha * k)
                node[pkeys..., n => s] = prob
            end
        end
        next!(p)
    end
    return BayesianNetwork(nodes, copy(dag.topology), copy(dag.A))
end

"""
    learn_parameters_em(dag::DirectAcyclicGraph, df::DataFrame; alpha = 0, max_iter = 100, tol = 1.0e-4, progress::Bool = isinteractive())

Estimate the CPTs of `dag` from data `df` that may contain `missing` entries, by
Expectation-Maximization, returning a fully-specified [`BayesianNetwork`](@ref) (call [`order!`](@ref)
before inference or sampling).

Starting from uniform CPTs, each iteration does:
- **E-step** — every row with missing values is expanded into all completions of its missing
  variables, each weighted by `P(missing | observed)` under the current network (from
  `joint_probability`); fully-observed rows keep weight `1`.
- **M-step** — the CPTs are re-estimated by the same counting as [`learn_parameters_mle`](@ref),
  summing these weights instead of counting rows.

Iteration stops when no CPT entry changes by more than `tol`, or after `max_iter` steps. `alpha` is
the Laplace/Dirichlet pseudo-count; node domains are the observed states plus any extra states
declared on the `dag`. With no missing values EM reduces exactly to [`learn_parameters_mle`](@ref).
Convergence is to a local optimum, so the (uniform) initialization matters. `dag` is left untouched.
`progress` shows a progress bar over the EM iterations (annotated with the current maximum CPT change),
defaulting to `isinteractive()` (REPL only); force it with `progress=true` / `progress=false`.

# Examples
```julia
dag = DirectAcyclicGraph()
add_node!(dag, :V)
add_node!(dag, :T; parents = [:V])

learned = learn_parameters_em(dag, df)   # df may contain `missing` entries
order!(learned)
```
"""
function learn_parameters_em(
        dag::DirectAcyclicGraph, df::DataFrame; alpha::Real = 0, max_iter::Int = 100, tol::Real = 1.0e-4, progress::Bool = isinteractive()
    )
    domains = Dict(n.name => sort(unique(vcat(get(dag.states, n.name, Symbol[]), collect(skipmissing(df[!, n.name]))))) for n in dag.nodes)
    bn = _em_uniform(dag, domains)
    p = Progress(max_iter; desc = "EM iteration ", enabled = progress)
    for _ in 1:max_iter
        newbn = _em_mstep(dag, _em_estep(dag, df, bn, domains), domains, alpha)
        change = _em_maxchange(bn, newbn)
        bn = newbn
        next!(p; showvalues = () -> [("max Δ", change)])
        if change < tol
            finish!(p)
            break
        end
    end
    return bn
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

# M-step: maximum-likelihood CPTs from the weighted, completed data (weights replace row counts).
function _em_mstep(dag::DirectAcyclicGraph, completed::DataFrame, domains, alpha)
    nodes = deepcopy(dag.nodes)
    weight = completed[!, :weight]
    for node in nodes
        n = node.name
        par = parents(dag, n)
        node_states = domains[n]
        k = length(node_states)
        for config in Iterators.product((domains[p] for p in par)...)
            pkeys = [par[i] => config[i] for i in eachindex(par)]
            mask = trues(nrow(completed))
            for (p, c) in zip(par, config)
                mask .&= completed[!, p] .== c
            end
            total = sum(weight[mask])
            for s in node_states
                cnt = sum(weight[mask .& (completed[!, n] .== s)])
                # clamp absorbs floating-point drift from summing fractional weights (cnt/total can
                # exceed 1 by an ulp, which the CPT's [0,1] check would reject)
                prob = total == 0 ? 1 / k : clamp((cnt + alpha) / (total + alpha * k), 0.0, 1.0)
                node[pkeys..., n => s] = prob
            end
        end
    end
    return BayesianNetwork(nodes, copy(dag.topology), copy(dag.A))
end

# Largest change in any CPT entry between successive iterations (rows are built in the same order).
_em_maxchange(a::BayesianNetwork, b::BayesianNetwork) = maximum(maximum(abs.(x.cpt.data.Π .- y.cpt.data.Π)) for (x, y) in zip(a.nodes, b.nodes))
