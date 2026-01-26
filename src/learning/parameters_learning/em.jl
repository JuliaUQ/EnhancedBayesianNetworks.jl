function learn_parameters_EM(df::DataFrame, net::BayesianNetwork2be, max_iter::Int64, states_space=nothing)
    if !issetequal(Symbol.(names(df)), net.nodes)
        error("nodes provided in the dataframe are not coherent with the ones provided in the network")
    end
    if isnothing(states_space)
        states_space = Dict(col => Symbol.(unique(df[!, col])) for col in Symbol.(names(df)))
    end
    return _bn_from_incomplete_df(df, net, max_iter, states_space)
end

function _bn_from_incomplete_df(df::DataFrame, net::BayesianNetwork2be, max_iter, states_space)
    complete_df = Symbol.(df[completecases(df), :])
    complete_cpts = _initialize_cpt(complete_df, net, states_space)
    nodes = map(complete_cpt -> DiscreteNode(complete_cpt[1], complete_cpt[2]), complete_cpts)
    incomplete_df = df[.!completecases(df), :]

    insertcols!(complete_df, :weight => fill(1, nrow(complete_df)))
    complete_df_updated = deepcopy(complete_df)
    bn = BayesianNetwork(nodes, net.topology, net.A)

    for i in range(1, max_iter)
        exploded_weighted_df = mapreduce(nt -> _expectation_step_single_missing_line(copy(nt), bn, states_space), vcat, eachrow(incomplete_df))
        complete_df_updated = vcat(complete_df_updated, exploded_weighted_df)
        new_nodes = map(n -> DiscreteNode(n.name, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(_counts_and_probs(complete_df_updated, n.name, parents(bn, n)[2], states_space)[2])), bn.nodes)
        bn = BayesianNetwork(new_nodes, bn.topology, bn.A)
        i += 1
    end
    return bn
end

function _expectation_step_single_missing_line(nt::NamedTuple, net::BayesianNetwork, states_space)
    exploded_nt = _explode_missing_single_line(nt, states_space)
    exploded_nt_counts = map(scenario -> joint_probability(net, Evidence(pairs(copy(scenario)))), eachrow(exploded_nt))
    exploded_nt_counts_normalized = exploded_nt_counts ./ sum(exploded_nt_counts)
    exploded_nt[!, :weight] = exploded_nt_counts_normalized
    return exploded_nt
end

function _explode_missing_single_line(nt::NamedTuple, states_space)
    ks = Symbol[]
    vs = Vector{Vector}()
    for (col, val) in zip(keys(nt), nt)
        if ismissing(val)
            push!(ks, col)
            push!(vs, states_space[col])
        else
            push!(ks, col)
            push!(vs, [Symbol(val)])
        end
    end
    combos = vec(Iterators.product(vs...) |> collect)
    return DataFrame(map(combo -> NamedTuple{Tuple(ks)}(combo), combos))
end

