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

S = DiscreteNode(:S, [:W])               # child of :W
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

states(node::DiscreteNode) = unique(node.cpt.data[:, node.name])

scenarios(node::DiscreteNode) = map(row -> [Symbol(col) => row[col] for col in names(node.cpt.data[:, Not("Π")])], eachrow(node.cpt.data[:, Not("Π")]))

isprecise(node::DiscreteNode) = all(isa.(node.cpt.data[:, :Π], Real))

isroot(node::DiscreteNode) = size(node.cpt.data, 2) == 2

parents(node::DiscreteNode) = Symbol.(names(node.cpt.data[:, Not(node.name, "Π")]))

function _inputs(node::DiscreteNode, evidence::Evidence)
    evstr = join(["$(repr(k)) => $(repr(v))" for (k, v) in evidence], ", ")
    if node.name ∉ keys(evidence)
        error("Invalid Evidence: evidence [$evstr] does not contain the node $(repr(node.name))")
    elseif values(evidence[node.name]) ∉ states(node)
        error("Invalid Evidence: evidence [$evstr] contains a not existing state $(repr(evidence[node.name])) for node $(repr(node.name))")
    elseif isempty(node.parameters)
        error("Invalid Node: node $(repr(node.name)) has an empty parameters dictionary")
    else
        idx = findfirst(p -> p.first == evidence[node.name], node.parameters)
        return node.parameters[idx].second
    end
end

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

function _extreme_nodes(node::DiscreteNode)
    function _extreme_points(cpt)
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
            par_comb = collect.(Iterators.product(map(p -> unique(node.cpt.data[:, p]), par)...))
            sub_cpts = map(pc -> filter(node.cpt, (par .=> pc)...), par_comb)
        else
            sub_cpts = [deepcopy(node.cpt.data)]
        end
        combinations = Iterators.product(map(sub_cpt -> _extreme_points(sub_cpt), sub_cpts)...) |> collect
        extreme_dataframes = vec(map(c -> vcat(c...), combinations))
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
function _extreme_probabilities(intervals::Vararg{Union{Real,Interval}})
    n = length(intervals)
    A = zeros(2 * n, n)
    A[collect(1:2:(2*n)), :] = Matrix(-1.0I, n, n)
    A[collect(2:2:(2*n)), :] = Matrix(1.0I, n, n)
    A = vcat(A, [-ones(n)'; ones(n)'])

    b = mapreduce(x -> flat(x), vcat, intervals)
    b[collect(1:2:(2*n))] = -b[collect(1:2:(2*n))]
    b = vcat(b, [-1 1]')

    h = mapreduce((Ai, bi) -> HalfSpace(Ai, bi), ∩, [A[i, :] for i in axes(A, 1)], b)
    v = doubledescription(h)
    return v.points.points
end