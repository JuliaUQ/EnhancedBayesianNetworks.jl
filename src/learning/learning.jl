"""
    DirectAcyclicGraph()

A directed acyclic graph: network **structure plus optional declared states**, without any
probabilities. It is the input to parameter learning ([`learn_parameters_mle`](@ref)) — declare the
nodes with [`add_node!`](@ref), wire the edges with [`add_child!`](@ref), then learn the CPTs from
data to obtain a fully-specified [`BayesianNetwork`](@ref).

Each node's domain is taken from the data at learn time; any states passed to [`add_node!`](@ref) are
*added* to that domain, letting a node keep states that never occur in the dataset (they end up in
the learned CPT with probability `0`). It is a standalone type (not an `AbstractNetwork`): it
supports [`add_node!`](@ref), [`add_child!`](@ref), [`parents`](@ref), and [`children`](@ref), but
none of the network operations that require CPTs (`order!`, `infer`, `sample`).

# Examples
```julia
dag = DirectAcyclicGraph()
add_node!(dag, :W, [:Foggy])                     # :Sunny/:Cloudy come from data; :Foggy guaranteed
add_node!(dag, :R; parents = [:W])               # domain taken entirely from the data
add_node!(dag, :P; parents = [:W])
add_node!(dag, :G; parents = [:R, :P])
add_child!(dag, :W, [:R, :P])
add_child!(dag, [:R, :P], :G)

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
    isempty(undefined) || error("Invalid DAG: parent(s) $undefined of $(repr(name)) are not defined; add them first")
    push!(dag.nodes, DiscreteNode(name, parents))
    idx = length(dag.nodes)
    dag.topology[name] = idx
    m = size(dag.A, 1)
    Anew = spzeros(Bool, m + 1, m + 1)
    Anew[1:m, 1:m] = dag.A
    dag.A = Anew
    isempty(parents) || (dag.A[getindex.(Ref(dag.topology), parents), idx] .= true)
    if !isempty(node_states)
        dag.states[name] = node_states
    end
    return dag
end

# Names of a node's parents / children, read from the adjacency matrix.
parents(dag::DirectAcyclicGraph, name::Symbol) = Symbol[dag.nodes[i].name for i in findnz(dag.A[:, dag.topology[name]])[1]]
children(dag::DirectAcyclicGraph, name::Symbol) = Symbol[dag.nodes[i].name for i in findnz(dag.A[dag.topology[name], :])[1]]

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
add_node!(dag, :V, [:maybe]); add_node!(dag, :T; parents = [:V])
add_child!(dag, :V, :T)

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

# include("parameters_learning/mle.jl")
# include("parameters_learning/em.jl")