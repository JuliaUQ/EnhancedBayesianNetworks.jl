"""
    DirectAcyclicGraph()

Network **structure plus optional declared states and estimators**, without any probabilities: the input to [`learn`](@ref). Nodes, their parents, their states and their estimator are declared with [`add_node!`](@ref), which wires the edges as it goes.

States passed to [`add_node!`](@ref) are *added* to the domain taken from the data, so a node keeps states that never occur in the dataset. An estimator may likewise be declared per node, letting one DAG mix precisely and imprecisely estimated nodes; it is all or nothing, and [`learn`](@ref) checks it.

It is a standalone type, not an `AbstractNetwork`: it supports [`add_node!`](@ref), [`parents`](@ref), [`children`](@ref) and `gplot`, but none of the network operations that need CPTs (`order!`, `infer`, `sample`).

# Examples

One estimator for the whole DAG, passed to [`learn`](@ref):

```julia
dag = DirectAcyclicGraph()
add_node!(dag, :W; states = [:Foggy])                            # :Foggy kept even if absent from data
add_node!(dag, :R; parents = [:W])                               # domain from data; edge :W -> :R wired here
add_node!(dag, :P; parents = [:W])
add_node!(dag, :G; parents = [:R, :P], states = [:mildly_wet])   # two parents, plus a declared state

learned = learn(dag, df, PreciseDirichletModel())                # -> BayesianNetwork
order!(learned)
```

Or one per node, declared on the DAG, where a single imprecise node makes the result a [`CredalNetwork`](@ref):

```julia
dag = DirectAcyclicGraph()
add_node!(dag, :W; states = [:Foggy], estimator = PreciseDirichletModel())
add_node!(dag, :R; parents = [:W], estimator = ImpreciseDirichletModel())
add_node!(dag, :P; parents = [:W], estimator = PreciseDirichletModel())
add_node!(dag, :G; parents = [:R, :P], estimator = PreciseDirichletModel())

learned = learn(dag, df)                                         # -> CredalNetwork
order!(learned)
```
"""
mutable struct DirectAcyclicGraph
    nodes::AbstractVector{DiscreteNode}
    topology::Dict
    A::SparseMatrixCSC
    states::Dict{Symbol, Vector{Symbol}}          # declared domain per node name
    estimators::Dict{Symbol, ParameterEstimator}  # declared estimator per node name
end

DirectAcyclicGraph() = DirectAcyclicGraph(
    DiscreteNode[], Dict{Symbol, Int}(), spzeros(Bool, 0, 0),
    Dict{Symbol, Vector{Symbol}}(), Dict{Symbol, ParameterEstimator}()
)

"""
    add_node!(dag::DirectAcyclicGraph, name::Symbol; states = Symbol[], parents = Symbol[], estimator = nothing)

Declare a node in a [`DirectAcyclicGraph`](@ref) by `name`. All three keywords are optional:

  - `parents` are the names of the node's parents, which become the conditioning columns of its CPT. Each must already be in the DAG, so nodes are added top-down, parents first; the edges are wired here. Omit it to declare a root.
  - `states` are *extra* domain states, kept even when they never appear in the training data. Under maximum likelihood they end up in the CPT with probability exactly `0`, otherwise with whatever the estimator assigns an unobserved state. Omit it to take the domain entirely from the data.
  - `estimator` is the [`ParameterEstimator`](@ref) to fit this node with, letting one DAG mix precise and imprecise nodes. It is all or nothing: [`learn`](@ref) rejects a DAG where some nodes declare one and others do not. Omit it throughout and pass the estimator to [`learn`](@ref) instead.

# Examples

```julia
dag = DirectAcyclicGraph()
add_node!(dag, :W; states = [:Foggy])     # root; :Sunny/:Cloudy from data, :Foggy guaranteed
add_node!(dag, :R; parents = [:W])        # edge :W -> :R created here
add_node!(dag, :G; parents = [:R])
```

Or with an estimator per node:

```julia
dag = DirectAcyclicGraph()
add_node!(dag, :W; states = [:Foggy], estimator = PreciseDirichletModel())
add_node!(dag, :R; parents = [:W], estimator = ImpreciseDirichletModel())
add_node!(dag, :G; parents = [:R], estimator = PreciseDirichletModel())
```
"""
function add_node!(
        dag::DirectAcyclicGraph,
        name::Symbol;
        states::Vector{Symbol} = Symbol[],
        parents::Vector{Symbol} = Symbol[],
        estimator::Union{Nothing, ParameterEstimator} = nothing
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
    if !isempty(states)
        dag.states[name] = states
    end
    if !isnothing(estimator)
        dag.estimators[name] = estimator
    end
    return dag
end

# Names of a node's parents / children, read from the adjacency matrix.
parents(dag::DirectAcyclicGraph, name::Symbol) = Symbol[dag.nodes[i].name for i in findnz(dag.A[:, dag.topology[name]])[1]]
children(dag::DirectAcyclicGraph, name::Symbol) = Symbol[dag.nodes[i].name for i in findnz(dag.A[dag.topology[name], :])[1]]
