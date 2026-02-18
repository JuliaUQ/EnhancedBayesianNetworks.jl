struct DiscreteNode <: AbstractDiscreteNode
    name::Symbol
    cpt::ConditionalProbabilityTable{DiscreteProbability}
    parameters::Vector{Pair{Symbol,Vector{Parameter}}}
    results::Dict{Vector{Symbol},Tuple}
    ## DiscreteNode without CPT
    function DiscreteNode(
        name::Symbol,
        parents::Vector{Symbol}=Symbol[],
        parameters::Vector{Pair{Symbol,Vector{Parameter}}}=Vector{Pair{Symbol,Vector{Parameter}}}(),
        results::Dict{Vector{Symbol},}=Dict{Vector{Symbol},Tuple}()
    )
        if name == :Π
            error(":Π is not allowed as node name")
        end
        cpt = ConditionalProbabilityTable{DiscreteProbability}([parents..., name])
        return new(name, cpt, parameters, results)
    end
    ## DiscreteNode with CPT
    function DiscreteNode(
        cpt::ConditionalProbabilityTable{DiscreteProbability},
        parameters::Vector{Pair{Symbol,Vector{Parameter}}}=Vector{Pair{Symbol,Vector{Parameter}}}(),
        results::Dict{Vector{Symbol},Tuple}=Dict{Vector{Symbol},Tuple}()
    )
        name = Symbol(names(cpt.data)[end-1])
        return new(name, cpt, parameters, results)
    end
end

DiscreteNode(name::Symbol, parameters::Vector{Pair{Symbol,Vector{Parameter}}}) = DiscreteNode(name, Symbol[], parameters, Dict{Vector{Symbol},Tuple}())

Base.setindex!(node::DiscreteNode, value, key...) = setindex!(node.cpt, value, key...)
Base.getindex(node::DiscreteNode, key...) = getindex(node.cpt, key...)

states(node::DiscreteNode) = unique(node.cpt.data[:, node.name])

scenarios(node::DiscreteNode) = map(row -> [Symbol(col) => row[col] for col in names(node.cpt.data[:, Not("Π")])], eachrow(node.cpt.data[:, Not("Π")]))

isprecise(node::DiscreteNode) = all(isa.(node.cpt.data[:, :Π], Real))

isroot(node::DiscreteNode) = size(node.cpt.data, 2) == 2

parents(node::DiscreteNode) = Symbol.(names(node.cpt.data[:, Not(node.name, "Π")]))

function _inputs(node::DiscreteNode, evidence::Evidence)
    if node.name ∉ keys(evidence)
        error("evidence `$evidence` does not contain the node `$(node.name)`")
    elseif values(evidence[node.name]) ∉ states(node)
        error("evidence `$evidence` contains a not existing state `$(evidence[node.name])` for node `$(node.name)`")
    elseif isempty(node.parameters)
        error("node `$(node.name)` dose not contain any parameters dictionary")
    else
        idx = findfirst(p -> p.first == evidence[node.name], node.parameters)
        return node.parameters[idx].second
    end
end

function _extreme_nodes(node::DiscreteNode)
    function _extreme_points(cpt)
        extreme_probs = EnhancedBayesianNetworks._extreme_probabilities(cpt.Π...)
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
        sub_cpts = groupby(node.cpt.data, names(node.cpt.data[:, Not(node.name, "Π")])) |> collect
        combinations = Iterators.product(map(sub_cpt -> _extreme_points(sub_cpt), sub_cpts)...) |> collect
        extreme_dataframes = vec(map(c -> vcat(c...), combinations))
        extreme_cpts = map(extreme_df -> ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}(extreme_df), extreme_dataframes)
        return map(extreme_cpt -> DiscreteNode(extreme_cpt, node.parameters, node.results), extreme_cpts)
    end
end

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