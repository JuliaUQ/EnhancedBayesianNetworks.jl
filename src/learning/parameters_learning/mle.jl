function learn_parameters_MLE(df::DataFrame, bn::BayesianNetwork2be, states_space=nothing)
    if !issetequal(Symbol.(names(df)), bn.nodes)
        error("nodes provided in the dataframe are not coherent with the ones provided in the network")
    end
    if isnothing(states_space)
        states_space = Dict(col => Symbol.(unique(df[!, col])) for col in Symbol.(names(df)))
    end
    cpts = _initialize_cpt(df, bn, states_space)
    nodes = map(cpt -> DiscreteNode(cpt[1], cpt[2]), cpts)
    return BayesianNetwork(nodes, bn.topology, bn.A)
end

function _initialize_cpt(df::DataFrame, net::BayesianNetwork2be, states_space)
    if !all(completecases(df))
        error("not complete DataFrame")
    end
    df = Symbol.(df)
    res = map(n -> (n, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(_counts_and_probs(df, n, parents(net, n)[2], states_space)[2])), net.nodes)
    return res
end

function _counts_and_probs(df::DataFrame, node::Symbol, pars::Vector{Symbol}, states_space)
    grouped = groupby(df, [pars...]) |> collect

    group_keys = map(sub_df -> NamedTuple{Tuple(pars)}(sub_df[1, pars]), grouped)
    evident_keys = Dict(k => Set{Any}() for k in keys(group_keys[1]))
    for tup in group_keys
        for (k, v) in pairs(tup)
            push!(evident_keys[k], v)
        end
    end
    evident_keys = Dict(k => collect(v) for (k, v) in evident_keys)
    parents_missing_states = _states_diff(evident_keys, states_space)
    if !isempty(parents_missing_states)
        error("provided dataframe does not contain the following states for the following nodes $parents_missing_states")
    end

    res = map(g -> _weighted_counts_and_probs_subgroup(g, states_space, node, pars, :weight), grouped)
    df_counts = mapreduce(t -> t[2], vcat, res)
    df_probs = mapreduce(t -> t[3], vcat, res)
    return df_counts, df_probs
end

function _weighted_counts_and_probs_subgroup(df::SubDataFrame, states_space, node::Symbol, pars::Vector{Symbol}, weights::Symbol)
    df_to_use = deepcopy(df)
    if weights ∉ Symbol.(names(df_to_use))
        insertcols!(df_to_use, weights => fill(1, nrow(df_to_use)))
    end
    group_key = NamedTuple{Tuple(pars)}(df_to_use[1, pars])
    counts = combine(groupby(df_to_use, node), weights => sum => :Π)
    missing_node_states = setdiff(states_space[node], unique(df_to_use[:, node]))
    map(ms -> push!(counts, (ms, 0)), missing_node_states)
    map(pv -> insertcols!(counts, 1, pv => fill(group_key[pv], nrow(counts))), keys(group_key))
    probs = deepcopy(counts)
    probs.Π = probs.Π ./ sum(probs.Π)
    return (group_key, counts, probs)
end

function _states_diff(states_space1::Dict, states_space2::Dict)
    diff = Dict()
    for k in keys(states_space1)
        if haskey(states_space2, k)
            v = setdiff(Set(states_space2[k]), Set(states_space1[k])) |> collect
            if !isempty(v)
                diff[k] = v
            end
        end
    end
    return diff
end