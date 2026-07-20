"""
    DiscreteNode(name, parents=Symbol[], parameters=[], results=nothing)
    DiscreteNode(cpt::ScenariosTable, parameters=[], results=nothing)

A discrete-state node. Holds a conditional probability table (`cpt`) over its parents' state
combinations and its own states. Optional per-state `parameters` (used when it feeds a
functional node) and optional stored `results` when a FunctionalNode is evaluated into a
DiscreteNode. Entries may be precise (`Real`) or imprecise/credal (`Interval`).
Build by name and fill with `node[parent1 => sₚ1, ..., name => sₙ] = p`.

# Examples
```julia
W = DiscreteNode(:W)                     # root node
W[:W => :sunny]  = 0.5
W[:W => :cloudy] = 0.5

# a node can carry per-state `parameters` (consumed when it feeds a functional node);
# each of its own states maps to a vector of `Parameter`s and is set at construction:
S = DiscreteNode(:S, [:W], [:on => [Parameter(0.5, :S)], :off => [Parameter(0.0, :S)]])
S[:W => :sunny,  :S => :on]  = 0.9
S[:W => :sunny,  :S => :off] = 0.1
S[:W => :cloudy, :S => :on]  = 0.2
S[:W => :cloudy, :S => :off] = 0.8

# credal (imprecise) entries use an Interval instead of a Real:
S[:W => :sunny, :S => :on] = Interval(0.8, 0.95)
```
"""
struct DiscreteNode <: AbstractDiscreteNode
    name::Symbol
    cpt::ScenariosTable{DiscreteProbability}
    parameters::Vector{Pair{Symbol,Vector{Parameter}}}
    results::Union{ScenariosTable{Any},Nothing}
    ## DiscreteNode without CPT
    function DiscreteNode(
        name::Symbol,
        parents::Vector{Symbol}=Symbol[],
        parameters::Vector{Pair{Symbol,Vector{Parameter}}}=Vector{Pair{Symbol,Vector{Parameter}}}(),
        results::Union{ScenariosTable{Any},Nothing}=nothing
    )
        if name == :Π
            error(":Π is not allowed as node name")
        end
        cpt = ScenariosTable{DiscreteProbability}([parents..., name], :Π)
        new(name, cpt, parameters, results)
    end
    ## DiscreteNode with CPT
    function DiscreteNode(
        cpt::ScenariosTable{DiscreteProbability},
        parameters::Vector{Pair{Symbol,Vector{Parameter}}}=Vector{Pair{Symbol,Vector{Parameter}}}(),
        results::Union{ScenariosTable{Any},Nothing}=nothing
    )
        name = Symbol(names(cpt.data)[end-1])
        new(name, cpt, parameters, results)
    end
end

DiscreteNode(name::Symbol, parameters::Vector{Pair{Symbol,Vector{Parameter}}}) = DiscreteNode(name, Symbol[], parameters, nothing)

Base.setindex!(node::DiscreteNode, value, key...) = setindex!(node.cpt, value, key...)

Base.getindex(node::DiscreteNode, key...) = getindex(node.cpt, key...)

"""
    states(node)

Return the vector of discrete states a node can take. Defined for discrete-type nodes:
a `DiscreteNode` returns its own states, a `DiscreteFunctionalNode` returns its two derived
states `:<name>_safe` and `:<name>_failed`. Continuous nodes have no discrete states, so
calling `states` on them is a `MethodError`.

# Examples
```julia
W = DiscreteNode(:W); W[:W => :sunny] = 0.5; W[:W => :cloudy] = 0.5
states(W)                                   # [:sunny, :cloudy]

m = Model(df -> df.x .^ 2, :y)
DF = DiscreteFunctionalNode(:DF, [:x], m, df -> 1 .- df.y)
states(DF)                                  # [:DF_safe, :DF_failed]
```
"""
states(node::DiscreteNode) = unique(node.cpt.data[:, node.name])

"""
    scenarios(node)

List every row of the node's CPT as a vector of `parent => state` pairs. For a `DiscreteNode`
each scenario also carries the node's own state; for a `ContinuousNode` only the parent-state
combinations are returned (a continuous node has no discrete state of its own). A root node
yields a single empty scenario. Not defined for functional nodes.

# Examples
```julia
S = DiscreteNode(:S, [:W])
S[:W => :sunny,  :S => :on]  = 0.9; S[:W => :sunny,  :S => :off] = 0.1
S[:W => :cloudy, :S => :on]  = 0.2; S[:W => :cloudy, :S => :off] = 0.8
scenarios(S)   # [[:W=>:sunny,:S=>:on], [:W=>:sunny,:S=>:off], [:W=>:cloudy,:S=>:on], [:W=>:cloudy,:S=>:off]]

C = ContinuousNode(:C, [:W]); C[:W => :sunny] = Normal(); C[:W => :cloudy] = Normal(2, 1)
scenarios(C)   # [[:W => :sunny], [:W => :cloudy]]
```
"""
scenarios(node::DiscreteNode) = map(row -> [Symbol(col) => row[col] for col in names(node.cpt.data[:, Not("Π")])], eachrow(node.cpt.data[:, Not("Π")]))

"""
    isprecise(node)

Return `true` if every entry in the node's CPT is precise, `false` if any is imprecise/credal.
For a `DiscreteNode` precise means every probability is a `Real` (imprecise entries are
`Interval`s); for a `ContinuousNode` precise means every entry is a `UnivariateDistribution`
(imprecise entries are `Interval`/`ProbabilityBox`).

# Examples
```julia
S = DiscreteNode(:S, [:W])
S[:W => :sunny,  :S => :on] = 0.9; S[:W => :sunny,  :S => :off] = 0.1
S[:W => :cloudy, :S => :on] = 0.2; S[:W => :cloudy, :S => :off] = 0.8
isprecise(S)                                # true

S[:W => :sunny, :S => :on] = Interval(0.8, 0.95)   # a credal entry
isprecise(S)                                # false

T = ContinuousNode(:T, Normal());  isprecise(T)     # true
```
"""
isprecise(node::DiscreteNode) = all(isa.(node.cpt.data[:, :Π], Real))

"""
    isroot(node)

Return `true` if the node has no parents (a root of the network). A `FunctionalNode` is never
a root (`false` always), since it is defined by models over its ancestors.

# Examples
```julia
W = DiscreteNode(:W); W[:W => :sunny] = 0.5; W[:W => :cloudy] = 0.5
isroot(W)                                   # true

S = DiscreteNode(:S, [:W]);  isroot(S)      # false
T = ContinuousNode(:T, Normal());  isroot(T)  # true
```
"""
isroot(node::DiscreteNode) = size(node.cpt.data, 2) == 2

"""
    parents(node)
    parents(net, name::Symbol)
    parents(net, node)

Return the parent names of a node as a `Vector{Symbol}` (empty for a root). The single-argument
form reads the parents off the node's own table (`DiscreteNode`/`ContinuousNode`); the
network forms look up the parents of `name` (or `node`) within `net`'s topology and work for
every node type, including functional nodes.

# Examples
```julia
W = DiscreteNode(:W); W[:W => :sunny] = 0.5; W[:W => :cloudy] = 0.5
S = DiscreteNode(:S, [:W])
parents(W)                                  # Symbol[]
parents(S)                                  # [:W]

bn = BayesianNetwork([W, S]); add_child!(bn, :W, :S); order!(bn)
parents(bn, :S)                             # [:W]
```
"""
parents(node::DiscreteNode) = Symbol.(names(node.cpt.data[:, Not(node.name, "Π")]))

"""
    sample(node::DiscreteNode, evidence::Evidence)
    sample(bn::BayesianNetwork, n::Int=1)

Draw discrete samples. Given a `DiscreteNode` and an `Evidence` fixing its parents (and,
optionally, the node itself), return one sampled state of the node; sampling a node whose entries
are imprecise raises an error. Given a `BayesianNetwork`, perform ancestral sampling of `n` joint
draws (ordering the network first) and return them as a `DataFrame` with one column per node.

# Examples
```julia
W = DiscreteNode(:W); W[:W => :sunny] = 0.5; W[:W => :cloudy] = 0.5
S = DiscreteNode(:S, [:W])
S[:W => :sunny,  :S => :on] = 0.9; S[:W => :sunny,  :S => :off] = 0.1
S[:W => :cloudy, :S => :on] = 0.2; S[:W => :cloudy, :S => :off] = 0.8

sample(S, Evidence(:W => :sunny))           # e.g. :on

bn = BayesianNetwork([W, S]); add_child!(bn, :W, :S); order!(bn)
sample(bn, 3)                               # 3×2 DataFrame with columns :W, :S
```
"""
function sample(node::DiscreteNode, evidence::Evidence)
    if node.name ∈ keys(evidence)
        return evidence[node.name]          # observed → return directly
    end
    if !isprecise(node)
        error("Sampling Error: cannot sample from imprecise node $(repr(node.name))")
    end
    parent_ev = filter(p -> p.first ∈ parents(node), evidence)
    cpt = filter(node.cpt, parent_ev...)
    if isempty(cpt)
        error("Sampling Error: evidence $(parent_ev) is not a valid configuration of parents $(parents(node)) for node $(repr(node.name))")
    end
    dist = Distributions.Categorical(Vector{Float64}(cpt.Π))
    return cpt[rand(dist), node.name]
end

function _inputs(node::DiscreteNode, evidence::Evidence)
    # Given evidence that fixes this node to one of its states, return the `Parameter`s attached to that state. 
    # Used to inject a discrete node's per-state parameters into a functional node's model.
    evstr = join(["$(repr(k)) => $(repr(v))" for (k, v) in evidence], ", ")
    # The node itself must be observed, its observed state must be valid, and it must carry parameters.
    if node.name ∉ keys(evidence)
        error("Invalid Evidence: evidence [$evstr] does not contain the node $(repr(node.name))")
    elseif values(evidence[node.name]) ∉ states(node)
        error("Invalid Evidence: evidence [$evstr] contains a not existing state $(repr(evidence[node.name])) for node $(repr(node.name))")
    elseif isempty(node.parameters)
        error("Invalid Node: node $(repr(node.name)) has an empty parameters dictionary")
    else
        # The node itself must be observed, its observed state must be valid, and it must carry parameters.
        idx = findfirst(p -> p.first == evidence[node.name], node.parameters)
        return node.parameters[idx].second
    end
end

function _extreme_nodes(node::DiscreteNode)
    # Expand a credal (imprecise) node into the finite set of precise DiscreteNodes sitting at the vertices of its credal set. 
    # A precise node is its own single vertex.
    function _extreme_points(cpt)
        # Vertices of one conditional distribution's credal set: for each extreme probability vector, copy the sub-CPT and overwrite its :Π column with that vector, giving a precise sub-CPT.
        extreme_probs = _extreme_probabilities(cpt.Π...)
        dfs = map(extreme_probs) do v
            df2 = copy(cpt)
            df2[!, :Π] = v
            df2
        end
        return dfs
    end

    if isprecise(node)
        return [node]
    else
        par = parents(node)
        if !isempty(par)
            # Each parent-state combination indexes an independent conditional distribution; split the CPT into those per-combination sub-CPTs.
            par_comb = collect.(Iterators.product(map(p -> unique(node.cpt.data[:, p]), par)...))
            sub_cpts = map(pc -> filter(node.cpt, (par .=> pc)...), par_comb)
        else
            # Root node: the whole CPT is a single conditional distribution.
            sub_cpts = [deepcopy(node.cpt.data)]
        end
        # Choose one vertex per sub-CPT independently; each combination reassembles a full precise CPT.
        combinations = Iterators.product(map(sub_cpt -> _extreme_points(sub_cpt), sub_cpts)...) |> collect
        extreme_dataframes = vec(map(c -> vcat(c...), combinations))
        # Materialize each precise CPT as a fresh DiscreteNode, carrying over parameters and results.
        extreme_nodes = DiscreteNode[]
        for df in extreme_dataframes
            vars = vcat(par, node.name)
            n = DiscreteNode(node.name, par, node.parameters, node.results)
            for row in eachrow(df)
                n[map(v->v=>row[v], vars)...] = row[:Π]
            end
            push!(extreme_nodes, n)
        end
        return extreme_nodes
    end
end

# TODO: Try to implement without PolyHedra
# Enumerate the vertices of the probability polytope for one conditional distribution:
# each pᵢ ∈ [lbᵢ, ubᵢ] and Σ pᵢ = 1. Returns the extreme probability vectors.
function _extreme_probabilities(intervals::Vararg{Union{Real,Interval}})
    n = length(intervals)
    # Box constraints as A x ≤ b: odd rows (-I) encode pᵢ ≥ lbᵢ, even rows (I) encode pᵢ ≤ ubᵢ.
    A = zeros(2 * n, n)
    A[collect(1:2:(2*n)), :] = Matrix(-1.0I, n, n)
    A[collect(2:2:(2*n)), :] = Matrix(1.0I, n, n)
    # Two extra rows encode the equality Σ pᵢ = 1 as the pair Σ pᵢ ≤ 1 and -Σ pᵢ ≤ -1.
    A = vcat(A, [-ones(n)'; ones(n)'])

    # Right-hand side: [lbᵢ, ubᵢ] per entry (a point [x, x] for a precise Real); negate the lb rows to match the -I rows, then append the sum bounds.
    b = mapreduce(x -> _flat(x), vcat, intervals)
    b[collect(1:2:(2*n))] = -b[collect(1:2:(2*n))]
    b = vcat(b, [-1 1]')

    # Build the H-representation (intersection of half-spaces) and enumerate its vertices.
    h = mapreduce((Ai, bi) -> HalfSpace(Ai, bi), ∩, [A[i, :] for i in axes(A, 1)], b)
    v = doubledescription(h)
    return v.points.points
end