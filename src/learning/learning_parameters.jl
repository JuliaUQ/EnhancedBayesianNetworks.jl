"""
    DirectAcyclicGraph()

A directed acyclic graph: network **structure plus optional declared states**, without any
probabilities. It is the input to parameter learning ([`learn_parameters_mle`](@ref)) — declare the
nodes and their parents with [`add_node!`](@ref) (edges are wired as you go), then learn the CPTs
from data to obtain a fully-specified [`BayesianNetwork`](@ref).

Each node's domain is taken from the data at learn time; any states passed to [`add_node!`](@ref) are
*added* to that domain, letting a node keep states that never occur in the dataset (they end up in
the learned CPT with probability `0`). It is a standalone type (not an `AbstractNetwork`): it
supports [`add_node!`](@ref), [`parents`](@ref), [`children`](@ref), and `gplot`, but none of the
network operations that require CPTs (`order!`, `infer`, `sample`).

# Examples
```julia
dag = DirectAcyclicGraph()
add_node!(dag, :W, [:Foggy])                     # :Sunny/:Cloudy come from data; :Foggy guaranteed
add_node!(dag, :R; parents = [:W])               # domain from data; edge :W -> :R wired here
add_node!(dag, :P; parents = [:W])
add_node!(dag, :G; parents = [:R, :P])

learned = learn_parameters_mle(dag, df)          # -> BayesianNetwork
order!(learned)
```
"""
mutable struct DirectAcyclicGraph
    nodes::AbstractVector{DiscreteNode}
    topology::Dict
    A::SparseMatrixCSC
    states::Dict{Symbol,Vector{Symbol}}   # declared domain per node name
end

DirectAcyclicGraph() = DirectAcyclicGraph(
    DiscreteNode[], Dict{Symbol,Int}(), spzeros(Bool, 0, 0), Dict{Symbol,Vector{Symbol}}()
)

"""
    add_node!(dag::DirectAcyclicGraph, name::Symbol, states=Symbol[]; parents=Symbol[])

Declare a node in a [`DirectAcyclicGraph`](@ref) by `name`, recording `parents` as its CPT columns
and wiring an edge from each parent (which must already be in the DAG — add nodes top-down, parents
first). `states` are *extra* domain states — states to keep in the node's domain even when they never
appear in the training data, so they end up in the learned CPT with probability `0`. Omit `states`
(or pass `[]`) to take the node's domain entirely from the data at learn time.

# Examples
```julia
dag = DirectAcyclicGraph()
add_node!(dag, :W, [:Foggy])              # root; :Sunny/:Cloudy from data, :Foggy guaranteed
add_node!(dag, :R; parents = [:W])        # edge :W -> :R created here
add_node!(dag, :G; parents = [:R])
```
"""
function add_node!(
    dag::DirectAcyclicGraph,
    name::Symbol,
    node_states::Vector{Symbol}=Symbol[];
    parents::Vector{Symbol}=Symbol[]
)
    if haskey(dag.topology, name)
        error("Invalid DAG: node $(repr(name)) is already present")
    end
    undefined = setdiff(parents, [n.name for n in dag.nodes])
    if !isempty(undefined)
        error("Invalid DAG: parent(s) $undefined of $(repr(name)) are not defined; add them first")
    end
    push!(dag.nodes, DiscreteNode(name, parents))
    idx = length(dag.nodes)
    dag.topology[name] = idx
    m = size(dag.A, 1)
    Anew = spzeros(Bool, m + 1, m + 1)
    Anew[1:m, 1:m] = dag.A
    dag.A = Anew
    if !isempty(parents)
        (dag.A[getindex.(Ref(dag.topology), parents), idx] .= true)
    end
    if !isempty(node_states)
        dag.states[name] = node_states
    end
    return dag
end

# Names of a node's parents / children, read from the adjacency matrix.
parents(dag::DirectAcyclicGraph, name::Symbol) = Symbol[dag.nodes[i].name for i in findnz(dag.A[:, dag.topology[name]])[1]]
children(dag::DirectAcyclicGraph, name::Symbol) = Symbol[dag.nodes[i].name for i in findnz(dag.A[dag.topology[name], :])[1]]

"""
    learn(dag::DirectAcyclicGraph, df::DataFrame; alpha=0, max_iter=100, tol=1e-4)

Learn the CPTs of `dag` from `df`, choosing the algorithm from the data: with no missing entries it
uses [`learn_parameters_mle`](@ref) (closed-form, exact); with any missing entries it uses
[`learn_parameters_em`](@ref). `alpha` is the Laplace/Dirichlet pseudo-count (both algorithms);
`max_iter` and `tol` control EM's convergence. Returns a fully-specified [`BayesianNetwork`](@ref);
call [`order!`](@ref) before inference or sampling. To force a specific algorithm, call
[`learn_parameters_mle`](@ref) or [`learn_parameters_em`](@ref) directly.

# Examples
```julia
learn(dag, df)                              # complete data -> MLE
learn(dag, df_with_missing)                 # has missing   -> EM
learn(dag, df; alpha = 1)                   # smoothing (either algorithm)
learn(dag, df_with_missing; tol = 1e-6, max_iter = 500)
```
"""
function learn(dag::DirectAcyclicGraph, df::DataFrame; alpha::Real=0, max_iter::Int=100, tol::Real=1e-4)
    incomplete = any(n -> any(ismissing, df[!, n.name]), dag.nodes)
    if incomplete
        return learn_parameters_em(dag, df; alpha=alpha, max_iter=max_iter, tol=tol)
    else
        return learn_parameters_mle(dag, df; alpha=alpha)
    end
end

"""
    learn_parameters_mle(dag::DirectAcyclicGraph, df::DataFrame; alpha=0)

Estimate the CPTs of `dag` from complete data `df` by maximum likelihood, returning a fully-specified
[`BayesianNetwork`](@ref) (call [`order!`](@ref) on it before inference or sampling).

Each node's domain is the states observed in `df` together with any extra states declared on the
`dag`, so declared-but-unobserved states appear with probability `0` (or `alpha`-smoothed mass). For
every node and every parent configuration, `P(node = s | parents = config)` is
`(count + alpha) / (total + alpha * k)`, with `alpha` a Laplace/Dirichlet pseudo-count (`alpha = 0`
is pure MLE, `k` the number of node states). A parent configuration absent from the data falls back
to a uniform distribution. `dag` is left untouched.

# Examples
```julia
dag = DirectAcyclicGraph()
add_node!(dag, :V, [:maybe])
add_node!(dag, :T; parents = [:V])

learned = learn_parameters_mle(dag, df)
order!(learned)
```
"""
function learn_parameters_mle(dag::DirectAcyclicGraph, df::DataFrame; alpha::Real=0)
    # domain of a node = states seen in the data ∪ extra states declared on the DAG
    statespace(col) = sort(unique(vcat(get(dag.states, col, Symbol[]), df[!, col])))
    nodes = deepcopy(dag.nodes)
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
                node[pkeys..., n=>s] = prob
            end
        end
    end
    return BayesianNetwork(nodes, copy(dag.topology), copy(dag.A))
end

"""
    learn_parameters_em(dag::DirectAcyclicGraph, df::DataFrame; alpha=0, max_iter=100, tol=1e-4)

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

# Examples
```julia
dag = DirectAcyclicGraph()
add_node!(dag, :V)
add_node!(dag, :T; parents = [:V])

learned = learn_parameters_em(dag, df)   # df may contain `missing` entries
order!(learned)
```
"""
function learn_parameters_em(dag::DirectAcyclicGraph, df::DataFrame; alpha::Real=0, max_iter::Int=100, tol::Real=1e-4)
    domains = Dict(n.name => sort(unique(vcat(get(dag.states, n.name, Symbol[]), collect(skipmissing(df[!, n.name]))))) for n in dag.nodes)
    bn = _em_uniform(dag, domains)
    for _ in 1:max_iter
        newbn = _em_mstep(dag, _em_estep(dag, df, bn, domains), domains, alpha)
        change = _em_maxchange(bn, newbn)
        bn = newbn
        if change < tol
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
                node[pkeys..., n=>s] = 1 / k
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
                node[pkeys..., n=>s] = prob
            end
        end
    end
    return BayesianNetwork(nodes, copy(dag.topology), copy(dag.A))
end

# Largest change in any CPT entry between successive iterations (rows are built in the same order).
_em_maxchange(a::BayesianNetwork, b::BayesianNetwork) = maximum(maximum(abs.(x.cpt.data.Π .- y.cpt.data.Π)) for (x, y) in zip(a.nodes, b.nodes))