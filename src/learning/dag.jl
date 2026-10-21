"""
    DirectAcyclicGraph()

A directed acyclic graph (DAG): network **structure plus optional declared states**, without any probabilities. It is the input to parameter learning. Nodes, their parents and their states are declared with [`add_node!`](@ref) (edges are wired as you go).

Each node's domain is taken from the data and the DAG at learn time; any states passed to [`add_node!`](@ref) are *added* to that domain, letting a node keep states that never occur in the dataset. A [`ParameterEstimator`](@ref) may likewise be declared per node, so that one DAG can mix precisely and imprecisely estimated nodes. It is a standalone type (not an `AbstractNetwork`): it supports [`add_node!`](@ref), [`parents`](@ref), [`children`](@ref), and `gplot`, but none of the network operations that require CPTs (`order!`, `infer`, `sample`).

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

Or one estimator per node, declared on the DAG. It is all or nothing, and a single imprecise node makes the result a [`CredalNetwork`](@ref):

```julia
dag = DirectAcyclicGraph()
add_node!(dag, :W; states = [:Foggy], estimator = PreciseDirichletModel())
add_node!(dag, :R; parents = [:W], estimator = ImpreciseDirichletModel())
add_node!(dag, :P; parents = [:W], estimator = PreciseDirichletModel())
add_node!(dag, :G; parents = [:R, :P], states = [:mildly_wet], estimator = PreciseDirichletModel())

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

  - `parents` are the names of the node's parents, which become the conditioning columns of its CPT. Each must already be in the DAG, so nodes have to be added top-down, parents first; an edge from each parent is wired here. Omit it to declare a root.
  - `states` are *extra* domain states: states to keep in the node's domain even when they never appear in the training data. They end up in the learned CPT with probability exactly `0` under maximum likelihood, and otherwise with whatever the chosen estimator assigns to an unobserved state. Omit it to take the node's domain entirely from the data.
  - `estimator` is the [`ParameterEstimator`](@ref) to fit this node with, letting one DAG mix precise and imprecise nodes. It is all or nothing: [`learn`](@ref) rejects a DAG where some nodes declare an estimator and others do not. Omit it throughout and pass the estimator to [`learn`](@ref) instead, which fits every node the same way.

# Examples

One estimator for the whole DAG, passed to [`learn`](@ref):

```julia
dag = DirectAcyclicGraph()
add_node!(dag, :W; states = [:Foggy])     # root; :Sunny/:Cloudy from data, :Foggy guaranteed
add_node!(dag, :R; parents = [:W])        # edge :W -> :R created here
add_node!(dag, :G; parents = [:R])
```

Or one estimator per node, declared here. It is all or nothing, and a single imprecise node makes the result a [`CredalNetwork`](@ref):

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
