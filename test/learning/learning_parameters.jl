@testitem "Verify columns" setup = [ExtraDeps] begin
    dag = DirectAcyclicGraph()
    add_node!(dag, :R)
    add_node!(dag, :G; parents = [:R])

    @test isnothing(EnhancedBayesianNetworks._verify_columns(dag, DataFrame(R = [:wet, :dry], G = [:green, :brown])))

    # a column of Union{Missing, Symbol} satisfies it too: whether `missing` is allowed is decided by `_verify_missingness` from the estimators, not here
    @test isnothing(EnhancedBayesianNetworks._verify_columns(dag, DataFrame(R = [:wet, missing], G = [:green, :brown])))

    # a node without a column is fatal, and every absent node is named at once
    @test_throws ErrorException("Invalid data: node(s) [:G] of the DAG have no column in the data") EnhancedBayesianNetworks._verify_columns(dag, DataFrame(R = [:wet]))
    @test_throws ErrorException("Invalid data: node(s) [:R, :G] of the DAG have no column in the data") (@suppress EnhancedBayesianNetworks._verify_columns(dag, DataFrame(X = [1])))

    # states are Symbols; a column of anything else is reported with its element type
    @test_throws ErrorException("Invalid data: node states are Symbols, but column(s) R::Int64 are not") EnhancedBayesianNetworks._verify_columns(dag, DataFrame(R = [1, 2], G = [:a, :b]))
    @test_throws ErrorException("Invalid data: node states are Symbols, but column(s) R::Int64, G::String are not") EnhancedBayesianNetworks._verify_columns(dag, DataFrame(R = [1, 2], G = ["a", "b"]))

    # the reverse direction is only a warning: a wide table may carry identifiers or measurements taken for another purpose, and no estimator ever reads them
    @test_logs (:warn, r"column\(s\) \[:Z\] that are not nodes of the DAG") EnhancedBayesianNetworks._verify_columns(dag, DataFrame(R = [:wet], G = [:green], Z = [9]))
end

@testitem "Verify missingness" setup = [ExtraDeps] begin
    dag = DirectAcyclicGraph()
    add_node!(dag, :R)
    add_node!(dag, :G; parents = [:R])
    complete = DataFrame(R = [:wet, :dry], G = [:green, :brown])
    incomplete = DataFrame(R = [:wet, missing], G = [:green, :brown])
    estimators(pairs...) = Dict{Symbol, ParameterEstimator}(pairs...)

    # complete data takes the ordinary path whatever the estimators, so even QuesenberryHurst, which is refused below, is fine here
    @test EnhancedBayesianNetworks._verify_missingness(dag, complete, estimators(:R => QuesenberryHurst(), :G => QuesenberryHurst())) === :complete

    # only the columns of DAG nodes are consulted: a `missing` in a column no estimator reads cannot change the treatment
    withjunk = DataFrame(R = [:wet, :dry], G = [:green, :brown], Z = [1, missing])
    @test (@suppress EnhancedBayesianNetworks._verify_missingness(dag, withjunk, estimators(:R => QuesenberryHurst(), :G => QuesenberryHurst()))) === :complete

    # the estimators fix the treatment, the caller does not choose it
    @test EnhancedBayesianNetworks._verify_missingness(dag, incomplete, estimators(:R => PreciseDirichletModel(), :G => PreciseDirichletModel(1))) === :em
    @test EnhancedBayesianNetworks._verify_missingness(dag, incomplete, estimators(:R => ImpreciseDirichletModel(), :G => ImpreciseDirichletModel(2))) === :envelope

    # QuesenberryHurst has no closed-form envelope over the completions of the data
    @test_throws ErrorException("Invalid estimator: column(s) [:R] contain missing values, which QuesenberryHurst on node(s) [:G] cannot be fitted from; its simultaneous confidence region has no closed-form envelope over the completions of the data, and bounding each entry separately would no longer carry the stated confidence. Use ImpreciseDirichletModel, whose bounds over the completions are exact") EnhancedBayesianNetworks._verify_missingness(dag, incomplete, estimators(:R => ImpreciseDirichletModel(), :G => QuesenberryHurst()))

    # expectation maximisation is a property of the whole network while bounding over the completions is local to a node, so a DAG that mixes them on incomplete data has no coherent treatment
    @test_throws ErrorException("Invalid DAG: column(s) [:R] contain missing values, and node(s) [:R] are estimated precisely while [:G] are not; expectation maximisation is a property of the whole network while bounding over the completions is local to a node, so the two treatments cannot be mixed. Declare one kind of estimator throughout, or complete the data") EnhancedBayesianNetworks._verify_missingness(dag, incomplete, estimators(:R => PreciseDirichletModel(), :G => ImpreciseDirichletModel()))

    # a DAG that is both mixed AND carries QuesenberryHurst reports the QuesenberryHurst refusal: it is checked first, so the message names the estimator that can never work rather than the combination, which the user could otherwise fix and still be refused
    @test_throws ErrorException("Invalid estimator: column(s) [:R] contain missing values, which QuesenberryHurst on node(s) [:G] cannot be fitted from; its simultaneous confidence region has no closed-form envelope over the completions of the data, and bounding each entry separately would no longer carry the stated confidence. Use ImpreciseDirichletModel, whose bounds over the completions are exact") EnhancedBayesianNetworks._verify_missingness(dag, incomplete, estimators(:R => PreciseDirichletModel(), :G => QuesenberryHurst()))
end

@testitem "EM uniform starting network" setup = [ExtraDeps] begin
    dag = DirectAcyclicGraph()
    add_node!(dag, :R)
    add_node!(dag, :G; parents = [:R])
    df = DataFrame(R = [:wet, :dry], G = [:green, :brown])
    domains = EnhancedBayesianNetworks._domains(dag, df)

    # EM has to start somewhere, and it starts knowing nothing: every conditional distribution is 1/k, for every parent configuration, so the first E-step weights all completions equally
    start = EnhancedBayesianNetworks._em_uniform(dag, domains)
    @test start isa BayesianNetwork
    @test all(all(n.cpt.data.Π .== 1 / length(domains[n.name])) for n in start.nodes)
    # :G has two states and two parent configurations, so four entries of 0.5
    @test length(start.nodes[2].cpt.data.Π) == 4

    # a network cannot differ from itself
    @test EnhancedBayesianNetworks._em_maxchange(start, start) == 0.0
end

@testitem "EM expectation step" setup = [ExtraDeps] begin
    dag = DirectAcyclicGraph()
    add_node!(dag, :R)
    add_node!(dag, :G; parents = [:R])
    complete = DataFrame(R = [:wet, :wet, :dry], G = [:green, :green, :brown])
    domains = EnhancedBayesianNetworks._domains(dag, complete)
    start = EnhancedBayesianNetworks._em_uniform(dag, domains)

    # on complete data the E-step is the identity apart from the weight column it appends
    kept = EnhancedBayesianNetworks._em_estep(dag, complete, start, domains)
    @test nrow(kept) == nrow(complete)
    @test all(kept.weight .== 1.0)
    @test :weight in propertynames(kept)

    # an incomplete row is replaced by one row per completion of its missing variables, so a row with one binary variable unknown becomes two. The weights of those completions sum to 1, which is what keeps the total weight equal to the number of original rows: the M-step then divides by a sample size that never changed, which is the whole reason EM stays normalised
    incomplete = DataFrame(R = [:wet, :wet, :dry, missing], G = [:green, :green, :brown, :brown])
    expanded = EnhancedBayesianNetworks._em_estep(dag, incomplete, start, domains)
    @test nrow(expanded) == 5
    @test sum(expanded.weight) ≈ nrow(incomplete)
    # under the uniform starting network the two completions of the ambiguous row are equally likely
    @test sort(expanded.weight) ≈ [0.5, 0.5, 1.0, 1.0, 1.0]

    # observed values survive the expansion: only the unknown variable varies across the new rows
    ambiguous = expanded[expanded.weight .< 1, :]
    @test all(ambiguous.G .== :brown)
    @test sort(ambiguous.R) == [:dry, :wet]
end

@testitem "EM expectation step leaves nothing incomplete" setup = [ExtraDeps] begin
    dag = DirectAcyclicGraph()
    add_node!(dag, :R)
    add_node!(dag, :P; parents = [:R])
    add_node!(dag, :G; parents = [:R, :P])
    # the fourth row has every variable unknown, the worst case the expansion has to survive
    df = DataFrame(
        R = [:wet, missing, :dry, missing, :wet],
        P = [:on, :on, missing, missing, :off],
        G = [:green, missing, :brown, missing, :green],
    )
    domains = EnhancedBayesianNetworks._domains(dag, df)
    completed = EnhancedBayesianNetworks._em_estep(dag, df, EnhancedBayesianNetworks._em_uniform(dag, domains), domains)

    # one row per assignment of the unknown variables: 1 + 4 + 2 + 8 + 1
    @test nrow(completed) == 16
    @test sum(completed.weight) ≈ nrow(df)

    # this is the invariant the weighted `_counts` depends on. It has no missing-data path of its own:
    # `isequal(missing, state)` is false for every state, so a surviving `missing` would contribute to
    # no state at all and `total` would silently fall below the weights of its configuration. The
    # E-step is what guarantees that cannot happen
    @test !any(any(ismissing, completed[!, c]) for c in names(completed))
    @test all(eltype(completed[!, c]) === Symbol for c in [:R, :P, :G])
end

@testitem "EM expectation step on an impossible row" setup = [ExtraDeps] begin
    dag = DirectAcyclicGraph()
    add_node!(dag, :R)
    add_node!(dag, :G; parents = [:R], states = [:never])
    train = DataFrame(R = [:wet, :dry], G = [:green, :brown])
    domains = EnhancedBayesianNetworks._domains(dag, train)

    # :never is declared on the DAG but absent from the data, so maximum likelihood gives it probability 0 in every parent configuration
    fitted = learn(dag, train, PreciseDirichletModel(0); progress = false)
    @test all(iszero, fitted.nodes[2].cpt.data[fitted.nodes[2].cpt.data.G .== :never, :Π])

    # a row the current network calls impossible would make every completion's weight 0 and the normalisation divide by zero. The E-step spreads such a row uniformly instead, so EM keeps the evidence rather than dropping it or producing NaN
    impossible = DataFrame(R = [missing], G = [:never])
    spread = EnhancedBayesianNetworks._em_estep(dag, impossible, fitted, domains)
    @test nrow(spread) == 2
    @test all(spread.weight .== 0.5)
    @test sum(spread.weight) ≈ 1.0
end

@testitem "EM iteration" setup = [ExtraDeps] begin
    dag = DirectAcyclicGraph()
    add_node!(dag, :R)
    add_node!(dag, :G; parents = [:R])
    estimators = Dict{Symbol, ParameterEstimator}(:R => PreciseDirichletModel(1), :G => PreciseDirichletModel(1))

    @test_throws ErrorException("Invalid call: max_iter must be at least 1, got 0") EnhancedBayesianNetworks._em(dag, DataFrame(R = [:wet], G = [:green]), Dict(:R => [:wet], :G => [:green]), estimators, 0, 1.0e-4, false)
    @test_throws ErrorException("Invalid call: tol must be strictly positive, got 0") EnhancedBayesianNetworks._em(dag, DataFrame(R = [:wet], G = [:green]), Dict(:R => [:wet], :G => [:green]), estimators, 10, 0, false)
    @test_throws ErrorException("Invalid call: tol must be strictly positive, got -1") EnhancedBayesianNetworks._em(dag, DataFrame(R = [:wet], G = [:green]), Dict(:R => [:wet], :G => [:green]), estimators, 10, -1, false)

    # on complete data the E-step is the identity, so the M-step sees the original counts and EM reproduces the ordinary fit exactly rather than approaching it
    complete = DataFrame(R = [:wet, :wet, :dry, :dry, :wet, :dry], G = [:green, :green, :brown, :brown, :brown, :green])
    domains = EnhancedBayesianNetworks._domains(dag, complete)
    iterated = EnhancedBayesianNetworks._em(dag, complete, domains, estimators, 100, 1.0e-10, false)
    direct = learn(dag, complete, PreciseDirichletModel(1); progress = false)
    @test all(a.cpt.data.Π == b.cpt.data.Π for (a, b) in zip(iterated, direct.nodes))

    # with data to disagree about `max_iter` genuinely bounds the loop rather than being ignored
    incomplete = DataFrame(R = [:wet, :wet, :dry, missing], G = [:green, :green, :brown, :brown])
    domains = EnhancedBayesianNetworks._domains(dag, incomplete)
    once = EnhancedBayesianNetworks._em(dag, incomplete, domains, estimators, 1, 1.0e-10, false)
    converged = EnhancedBayesianNetworks._em(dag, incomplete, domains, estimators, 100, 1.0e-10, false)
    @test any(a.cpt.data.Π != b.cpt.data.Π for (a, b) in zip(once, converged))

    # and once converged, more iterations change nothing
    longer = EnhancedBayesianNetworks._em(dag, incomplete, domains, estimators, 500, 1.0e-10, false)
    @test all(a.cpt.data.Π ≈ b.cpt.data.Π for (a, b) in zip(converged, longer))
end

@testitem "_domains fucntion" setup = [ExtraDeps] begin
    dag = DirectAcyclicGraph()
    add_node!(dag, :R)
    add_node!(dag, :G; parents = [:R])

    # every node gets an entry, and the domain is the states observed in the data
    domains = EnhancedBayesianNetworks._domains(dag, DataFrame(R = [:wet, :dry, :wet], G = [:green, :brown, :green]))
    @test domains isa Dict{Symbol, Vector{Symbol}}
    @test sort(collect(keys(domains))) == [:G, :R]

    # the union is SORTED, which is what makes `Iterators.product` enumerate parent configurations in the same order on every run.
    @test domains[:R] == [:dry, :wet]
    @test EnhancedBayesianNetworks._domains(dag, DataFrame(R = [:dry, :wet], G = [:green, :brown]))[:R] == [:dry, :wet]

    @test EnhancedBayesianNetworks._domains(dag, DataFrame(R = [:wet, missing, :dry], G = [:green, :brown, missing]))[:R] == [:dry, :wet]

    # states declared through `add_node!` are ADDED to those observed, which is how a node keeps a state the training data never shows
    declared = DirectAcyclicGraph()
    add_node!(declared, :R; states = [:foggy])
    add_node!(declared, :G; parents = [:R], states = [:zzz, :aaa])
    domains = EnhancedBayesianNetworks._domains(declared, DataFrame(R = [:wet, :dry], G = [:green, :brown]))
    @test domains[:R] == [:dry, :foggy, :wet]
    @test domains[:G] == [:aaa, :brown, :green, :zzz]

    # declaring a state that also occurs in the data does not duplicate it
    domains = EnhancedBayesianNetworks._domains(declared, DataFrame(R = [:wet, :foggy], G = [:green, :aaa]))
    @test domains[:R] == [:foggy, :wet]
    @test domains[:G] == [:aaa, :green, :zzz]

    # a declared state survives even when the column holds nothing else
    @test EnhancedBayesianNetworks._domains(declared, DataFrame(R = [missing, missing], G = [:green, :brown]))[:R] == [:foggy]

    @test isempty(EnhancedBayesianNetworks._domains(dag, DataFrame(R = [missing, missing], G = [:green, :brown]))[:R])
end

@testitem "Counts on complete data" setup = [ExtraDeps] begin
    df = DataFrame(
        R = [:wet, :wet, :wet, :dry, :dry],
        G = [:green, :green, :brown, :green, :brown],
    )
    fields(c) = (c.counts, c.total, c.node_missing, c.parents_missing, c.both_missing)

    # a root has no parent configuration to match, so the one configuration covers every row
    @test fields(EnhancedBayesianNetworks._counts(df, :G, [:green, :brown], Pair{Symbol, Symbol}[], false)) == ([3, 2], 5, 0, [0, 0], 0)

    # `total` is `sum(counts)` by construction
    for config in ([:R => :wet], [:R => :dry], Pair{Symbol, Symbol}[])
        counted = EnhancedBayesianNetworks._counts(df, :G, [:green, :brown], config, false)
        @test counted.total == sum(counted.counts)
    end

    @test fields(EnhancedBayesianNetworks._counts(df, :G, [:green, :brown], [:R => :wet], false)) == ([2, 1], 3, 0, [0, 0], 0)

    # a state declared on the DAG but never observed still gets an entry, counted as 0, which is what lets `add_node!`'s `states` keyword survive into the CPT
    @test fields(EnhancedBayesianNetworks._counts(df, :G, [:green, :brown, :never], Pair{Symbol, Symbol}[], false)) == ([3, 2, 0], 5, 0, [0, 0, 0], 0)

    # a configuration no row matches gives all zeros rather than an error
    @test fields(EnhancedBayesianNetworks._counts(df, :G, [:green, :brown], [:R => :foggy], false)) == ([0, 0], 0, 0, [0, 0], 0)
end

@testitem "Counts on incomplete data" setup = [ExtraDeps] begin
    # twelve rows over R -> G <- P with `missing` in every combination that matters
    df = DataFrame(
        R = [:wet, :wet, :wet, :wet, :wet, missing, missing, :wet, missing, :dry, :dry, :wet],
        P = [:on, :on, :on, :on, missing, :on, missing, missing, missing, :on, missing, :off],
        G = [:green, :green, :brown, missing, :green, :brown, :green, missing, missing, :green, :green, :green],
    )
    states = [:green, :brown]
    fields(c) = (c.counts, c.total, c.node_missing, c.parents_missing, c.both_missing)
    counts(config) = fields(EnhancedBayesianNetworks._counts(df, :G, states, config, true))

    # rows 1-3 are complete, row 4 has G unknown,
    # rows 5-7 could be completed as this configuration with G observed,
    # rows 8-9 with G unknown too,
    # row 11 is EXCLUDED although P is missing, because R is observed as :dry and contradicts :wet: one contradicting observation is enough, however much else is unknown
    @test counts([:R => :wet, :P => :on]) == ([2, 1], 3, 1, [2, 1], 2)

    # the same rows classify differently per configuration: row 11 now fits, rows 1-4 drop out
    @test counts([:R => :dry, :P => :on]) == ([1, 0], 1, 0, [2, 1], 1)

    # a configuration with no complete row at all still carries virtual frequencies
    @test counts([:R => :dry, :P => :off]) == ([0, 0], 0, 0, [2, 0], 1)

    # conditioning on one parent only: the rows whose :R is unknown become the ambiguous ones
    @test counts([:R => :wet]) == ([4, 1], 5, 2, [1, 1], 1)

    # a root has no parent that could be missing, so nothing is ever ambiguous and only the rows whose own value is unknown are recorded
    @test counts(Pair{Symbol, Symbol}[]) == ([7, 2], 9, 3, [0, 0], 0)

    # `total` still excludes every incomplete row, so it remains `sum(counts)`
    for config in ([:R => :wet, :P => :on], [:R => :dry, :P => :off], Pair{Symbol, Symbol}[])
        counted = EnhancedBayesianNetworks._counts(df, :G, states, config, true)
        @test counted.total == sum(counted.counts)
    end

    # the flag is a switch: with it false the same frame yields the same `counts` and `total`
    cheap = EnhancedBayesianNetworks._counts(df, :G, states, [:R => :wet, :P => :on], false)
    @test fields(cheap) == ([2, 1], 3, 0, [0, 0], 0)
end

@testitem "Counts from EM weights" setup = [ExtraDeps] begin
    df = DataFrame(R = [:wet, :wet, :dry, :wet], G = [:green, :brown, :green, :green])
    weights = [1.0, 0.5, 0.5, 2.0]

    counted = EnhancedBayesianNetworks._counts(df, :G, [:green, :brown], [:R => :wet], weights)
    @test counted isa EnhancedBayesianNetworks.ConfigurationCounts{Float64}
    @test counted.counts == [3.0, 0.5]
    @test counted.total == 3.5
    @test counted.node_missing == 0.0
    @test counted.parents_missing == [0.0, 0.0]
    @test counted.both_missing == 0.0

    # all-ones weights reproduce the unweighted counts exactly, so the M-step on complete data is the ordinary fitting walk
    plain = EnhancedBayesianNetworks._counts(df, :G, [:green, :brown], [:R => :wet], ones(nrow(df)))
    unweighted = EnhancedBayesianNetworks._counts(df, :G, [:green, :brown], [:R => :wet], false)
    @test plain.counts == unweighted.counts
    @test plain.total == unweighted.total
end

@testitem "_uniform_estimators function" setup = [ExtraDeps] begin
    dag = DirectAcyclicGraph()
    add_node!(dag, :W)
    add_node!(dag, :P; parents = [:W])
    add_node!(dag, :R; parents = [:W])
    add_node!(dag, :G; parents = [:R, :P])

    spread = EnhancedBayesianNetworks._uniform_estimators(dag, ImpreciseDirichletModel(2))
    @test spread isa Dict{Symbol, ParameterEstimator}
    @test sort(collect(keys(spread))) == [:G, :P, :R, :W]
    @test all(e == ImpreciseDirichletModel(2) for e in values(spread))
    @test spread isa AbstractDict{Symbol, <:ParameterEstimator}

    # estimators already declared on the DAG are ignored rather than merged: `learn` refuses that
    # combination before reaching here, so this function never has to choose between them
    declared = DirectAcyclicGraph()
    add_node!(declared, :A; estimator = QuesenberryHurst())
    add_node!(declared, :B; parents = [:A], estimator = PreciseDirichletModel(5))
    overridden = EnhancedBayesianNetworks._uniform_estimators(declared, PreciseDirichletModel(1))
    @test all(e == PreciseDirichletModel(1) for e in values(overridden))
    @test declared.estimators[:A] == QuesenberryHurst()
    @test declared.estimators[:B] == PreciseDirichletModel(5)
end

@testitem "Fit loop" setup = [ExtraDeps] begin
    dag = DirectAcyclicGraph()
    add_node!(dag, :R)
    add_node!(dag, :G; parents = [:R], states = [:never])
    df = DataFrame(R = [:wet, :wet, :dry, :dry, :wet, :dry], G = [:green, :green, :brown, :brown, :brown, :green])
    domains = EnhancedBayesianNetworks._domains(dag, df)
    estimators = Dict{Symbol, ParameterEstimator}(:R => PreciseDirichletModel(0), :G => ImpreciseDirichletModel(1))

    fitted = EnhancedBayesianNetworks._fit_loop(dag, df, estimators, domains)
    @test fitted isa Vector{DiscreteNode}
    @test [n.name for n in fitted] == [:R, :G]

    # Fitting never writes into the DAG it was given and the same
    @test all(isempty(n.cpt.data.Π) for n in dag.nodes)

    @test isprecise(fitted[1])
    @test !isprecise(fitted[2])

    # every parent configuration gets an entry for every state, including the state declared through `add_node!` that the data never shows
    @test nrow(fitted[2].cpt.data) == length(domains[:R]) * length(domains[:G])
    @test :never in fitted[2].cpt.data.G

    # `has_missing` is the switch `_fit_cpts` sets from the data: on a complete frame it changes nothing, because the virtual frequencies it enables are all zero
    @test EnhancedBayesianNetworks._fit_loop(dag, df, estimators, domains; has_missing = true)[2].cpt.data.Π == fitted[2].cpt.data.Π

    # `weights` replaces row counting by summing responsibilities, so all-ones weights reproduce the unweighted fit exactly
    precise = Dict{Symbol, ParameterEstimator}(:R => PreciseDirichletModel(1), :G => PreciseDirichletModel(1))
    @test EnhancedBayesianNetworks._fit_loop(dag, df, precise, domains; weights = ones(nrow(df)))[2].cpt.data.Π ==
        EnhancedBayesianNetworks._fit_loop(dag, df, precise, domains)[2].cpt.data.Π
end

@testitem "Fit CPTs dispatch" setup = [ExtraDeps] begin
    dag = DirectAcyclicGraph()
    add_node!(dag, :R)
    add_node!(dag, :G; parents = [:R])
    complete = DataFrame(R = [:wet, :wet, :dry, :dry, :wet, :dry], G = [:green, :green, :brown, :brown, :brown, :green])
    incomplete = DataFrame(R = [:wet, :wet, :dry, missing, :wet, :dry], G = [:green, :green, :brown, :brown, missing, :green])
    precise = Dict{Symbol, ParameterEstimator}(:R => PreciseDirichletModel(1), :G => PreciseDirichletModel(1))
    imprecise = Dict{Symbol, ParameterEstimator}(:R => ImpreciseDirichletModel(), :G => ImpreciseDirichletModel())

    @test_throws ErrorException("Invalid data: the data frame has no rows, there is nothing to learn from") EnhancedBayesianNetworks._fit_cpts(dag, DataFrame(R = Symbol[], G = Symbol[]), precise)

    # a node whose column is entirely missing has no states at all, which neither fitting path can work with, so it is refused here rather than failing inside one of them
    @test_throws ErrorException("Invalid data: node(s) [:R] have no states; their column is entirely missing and none were declared through add_node!") EnhancedBayesianNetworks._fit_cpts(dag, DataFrame(R = [missing, missing], G = [:green, :brown]), precise)

    # the three treatments are chosen from the data and the estimators, never from a keyword
    domains = EnhancedBayesianNetworks._domains(dag, complete)
    @test EnhancedBayesianNetworks._fit_cpts(dag, complete, precise)[2].cpt.data.Π == EnhancedBayesianNetworks._fit_loop(dag, complete, precise, domains)[2].cpt.data.Π

    domains = EnhancedBayesianNetworks._domains(dag, incomplete)
    @test EnhancedBayesianNetworks._fit_cpts(dag, incomplete, imprecise)[2].cpt.data.Π == EnhancedBayesianNetworks._fit_loop(dag, incomplete, imprecise, domains; has_missing = true)[2].cpt.data.Π

    @test EnhancedBayesianNetworks._fit_cpts(dag, incomplete, precise)[2].cpt.data.Π == EnhancedBayesianNetworks._em(dag, incomplete, domains, precise, 100, 1.0e-4, false)[2].cpt.data.Π

    # `max_iter` and `tol` reach the EM path and nowhere else
    early = EnhancedBayesianNetworks._fit_cpts(dag, incomplete, precise; max_iter = 1)
    @test any(a.cpt.data.Π != b.cpt.data.Π for (a, b) in zip(early, EnhancedBayesianNetworks._fit_cpts(dag, incomplete, precise)))
    @test EnhancedBayesianNetworks._fit_cpts(dag, complete, precise; max_iter = 1)[2].cpt.data.Π == EnhancedBayesianNetworks._fit_cpts(dag, complete, precise)[2].cpt.data.Π
end

@testitem "Bounds over every completion of the data" setup = [ExtraDeps] begin
    # An independent test for the robust Bayesian estimator: enumerate EVERY completion of the  incomplete frame, apply the plain IDM interval to each, and take the envelope. The closed form  in `_estimate` must reproduce it exactly.
    function envelope(df, node, states, parent_names, config, s)
        columns = [node; parent_names]
        domains = Dict(c => sort(unique(skipmissing(df[!, c]))) for c in columns)
        options = [Iterators.product((ismissing(df[i, c]) ? domains[c] : [df[i, c]] for c in columns)...) for i in 1:nrow(df)]
        lower = fill(Inf, length(states))
        upper = fill(-Inf, length(states))
        for completion in Iterators.product(options...)
            selected = [all(completion[i][1 + j] == last(config[j]) for j in eachindex(config)) for i in 1:nrow(df)]
            total = count(selected)
            for (k, state) in enumerate(states)
                cnt = count(i -> selected[i] && completion[i][1] == state, 1:nrow(df))
                lower[k] = min(lower[k], cnt / (total + s))
                upper[k] = max(upper[k], (cnt + s) / (total + s))
            end
        end
        return [Interval(lower[k], upper[k]) for k in eachindex(states)]
    end

    # :G has THREE states on purpose. At k = 2 the lower virtual frequency `ambiguous - parents_missing[i]`
    # is just the other state's count, so an implementation summing only one other state would pass;
    # from k = 3 the sum over the remaining states is a genuinely different quantity
    df = DataFrame(
        R = [:wet, :wet, :dry, missing, :wet, missing, :dry, :wet],
        P = [:on, missing, :on, :on, missing, missing, :off, :on],
        G = [:green, :brown, :brown, missing, :yellow, :green, :yellow, missing],
    )
    states = [:green, :brown, :yellow]

    for config in ([:R => :wet, :P => :on], [:R => :dry, :P => :on], [:R => :wet, :P => :off]), s in (1.0, 2.0)
        counted = EnhancedBayesianNetworks._counts(df, :G, states, config, true)
        @test EnhancedBayesianNetworks._estimate(ImpreciseDirichletModel(s), counted) == envelope(df, :G, states, [:R, :P], config, s)
    end

    # with one parent, and on a root, where every row is in the single configuration
    for config in ([:R => :wet], [:R => :dry]), s in (1.0, 2.0)
        counted = EnhancedBayesianNetworks._counts(df, :G, states, config, true)
        @test EnhancedBayesianNetworks._estimate(ImpreciseDirichletModel(s), counted) == envelope(df, :G, states, [:R], config, s)
    end
end

@testitem "Learn" setup = [ExtraDeps] begin
    # V: 8 :yes, 8 :no | T|V=:yes: 4 :t1, 4 :t2 | T|V=:no: 2 :t1, 6 :t2, so every estimate is exact
    df = DataFrame(
        V = [fill(:yes, 8); fill(:no, 8)],
        T = [fill(:t1, 4); fill(:t2, 4); fill(:t1, 2); fill(:t2, 6)],
    )
    builddag() = (d = DirectAcyclicGraph(); add_node!(d, :V); add_node!(d, :T; parents = [:V]); d)
    getp(net, sc...) = net.nodes[net.topology[last(sc).first]][sc...]

    learned = learn(builddag(), df, PreciseDirichletModel(); progress = false)
    @test learned isa BayesianNetwork
    order!(learned)
    @test getp(learned, :V => :yes) == 0.5
    @test getp(learned, :V => :yes, :T => :t1) == 0.5
    @test getp(learned, :V => :no, :T => :t1) == 0.25
    @test getp(learned, :V => :no, :T => :t2) == 0.75

    # Laplace smoothing: (cnt + α) / (total + α * k)
    smoothed = learn(builddag(), df, PreciseDirichletModel(1); progress = false)
    @test getp(smoothed, :V => :no, :T => :t1) == (2 + 1) / (8 + 1 * 2)

    # an imprecise estimator gives a CredalNetwork over the same data
    @test learn(builddag(), df, ImpreciseDirichletModel(); progress = false) isa CredalNetwork
    @test learn(builddag(), df, QuesenberryHurst(); progress = false) isa CredalNetwork

    # a state declared on the DAG but never observed stays at probability 0 under maximum likelihood
    declared = DirectAcyclicGraph()
    add_node!(declared, :V)
    add_node!(declared, :T; parents = [:V], states = [:t3])
    @test getp(learn(declared, df, PreciseDirichletModel(); progress = false), :V => :yes, :T => :t3) == 0.0

    # a parent configuration that never occurs falls back to the uniform distribution
    unobserved = DirectAcyclicGraph()
    add_node!(unobserved, :V; states = [:maybe])
    add_node!(unobserved, :T; parents = [:V])
    fallback = learn(unobserved, df, PreciseDirichletModel(); progress = false)
    @test getp(fallback, :V => :maybe) == 0.0
    @test getp(fallback, :V => :maybe, :T => :t1) == 0.5

    # the two forms exclude each other: an estimator passed to a DAG that declares its own is a conflict with no sensible winner, and a DAG declaring estimators for only some nodes is refused at learn time rather than at `add_node!`, since a DAG under construction is legitimately partial
    mixed = DirectAcyclicGraph()
    add_node!(mixed, :V; estimator = PreciseDirichletModel())
    add_node!(mixed, :T; parents = [:V], estimator = ImpreciseDirichletModel())
    @test_throws ErrorException("Invalid call: node(s) [:V, :T] declare their own estimator; call learn(dag, df) to fit with the declared ones, or build the DAG without estimators to fit every node with a single one") learn(mixed, df, PreciseDirichletModel(); progress = false)
    @test learn(mixed, df; progress = false) isa CredalNetwork

    @test_throws ErrorException("Invalid DAG: no node declares an estimator; declare one for every node with add_node!, or pass a single estimator to learn") learn(builddag(), df; progress = false)

    partial = DirectAcyclicGraph()
    add_node!(partial, :V; estimator = PreciseDirichletModel())
    add_node!(partial, :T; parents = [:V])
    @test_throws ErrorException("Invalid DAG: node(s) [:T] declare no estimator while [:V] do; declare one for every node or none at all") learn(partial, df; progress = false)

    # a node whose column is entirely missing has no states at all and is refused, unless `add_node!` supplied them, in which case the node is simply certain
    blank = DataFrame(V = [missing, missing], T = [:t1, :t2])
    @test_throws ErrorException("Invalid data: node(s) [:V] have no states; their column is entirely missing and none were declared through add_node!") learn(builddag(), blank, PreciseDirichletModel(); progress = false)
    rescued = DirectAcyclicGraph()
    add_node!(rescued, :V; states = [:foggy])
    add_node!(rescued, :T; parents = [:V])
    @test learn(rescued, blank, PreciseDirichletModel(); progress = false) isa BayesianNetwork
end

@testitem "Learn recovers a network it was sampled from" setup = [ExtraDeps, SetupAsiaBN] begin
    builddag() = begin
        d = DirectAcyclicGraph()
        add_node!(d, :V)
        add_node!(d, :S)
        add_node!(d, :T; parents = [:V])
        add_node!(d, :L; parents = [:S])
        add_node!(d, :B; parents = [:S])
        add_node!(d, :E; parents = [:L, :T])
        add_node!(d, :D; parents = [:B, :E])
        add_node!(d, :X; parents = [:E])
        d
    end
    getp(net, sc...) = net.nodes[net.topology[last(sc).first]][sc...]
    # well-sampled scenarios, so the comparison is not dominated by noise in rare configurations
    scenarios = [
        (:V => :YesV,), (:S => :YesS,),
        (:V => :YesV, :T => :YesT), (:V => :NoV, :T => :YesT),
        (:S => :YesS, :B => :YesB), (:S => :NoS, :B => :YesB),
        (:S => :YesS, :L => :YesL), (:S => :NoS, :L => :YesL),
        (:B => :YesB, :E => :YesE, :D => :YesD), (:B => :NoB, :E => :NoE, :D => :YesD),
        (:E => :YesE, :X => :YesX), (:E => :NoE, :X => :YesX),
    ]

    # the end-to-end check nothing else makes: sample a large dataset from a network whose CPTs are known, learn them back, and compare. It exercises the whole walk at once, over eight nodes with up to two parents, and would catch a CPT written under the wrong parent configuration, which every unit test above is blind to because each checks one configuration in isolation
    df = sample(bn, 50000)
    learned = learn(builddag(), df, PreciseDirichletModel(); progress = false)
    @test learned isa BayesianNetwork
    order!(learned)
    @test all(abs(getp(learned, sc...) - getp(bn, sc...)) < 0.06 for sc in scenarios)

    # and the learned network is usable downstream, which no other test here confirms
    @test infer(learned, :X, Evidence(:V => :YesV)) isa EnhancedBayesianNetworks.Posterior

    # the same through expectation maximisation, with about one cell in seven blanked
    blanked = allowmissing(sample(bn, 3000))
    for (j, c) in enumerate(names(blanked)), i in 1:nrow(blanked)
        (i + j) % 7 == 0 && (blanked[i, c] = missing)
    end
    recovered = learn(builddag(), blanked, PreciseDirichletModel(); progress = false)
    @test recovered isa BayesianNetwork
    order!(recovered)
    common = [
        (:V => :YesV,), (:V => :NoV,), (:S => :YesS,), (:S => :NoS,),
        (:V => :NoV, :T => :YesT), (:S => :NoS, :B => :YesB), (:S => :NoS, :L => :YesL),
        (:B => :YesB, :E => :NoE, :D => :YesD), (:B => :NoB, :E => :NoE, :D => :YesD),
        (:E => :NoE, :X => :YesX),
    ]
    @test all(abs(getp(recovered, sc...) - getp(bn, sc...)) < 0.05 for sc in common)
end

@testitem "Learn a credal network and use it" setup = [ExtraDeps] begin
    # three states and three parents, neither of which any other test here reaches: at k = 2 the lower virtual frequency degenerates, and with two parents the configuration product is small enough to hide an ordering mistake in the CPT key prefixes
    dag = DirectAcyclicGraph()
    add_node!(dag, :W)
    add_node!(dag, :R; parents = [:W])
    add_node!(dag, :P; parents = [:W])
    add_node!(dag, :G; parents = [:W, :R, :P])

    df = DataFrame(
        W = repeat([:warm, :cold], inner = 8),
        R = repeat([:wet, :dry], inner = 4, outer = 2),
        P = repeat([:on, :off], inner = 2, outer = 4),
        G = [
            :green, :brown, :yellow, :green, :brown, :green, :green, :yellow,
            :yellow, :brown, :green, :green, :brown, :yellow, :green, :brown,
        ],
    )

    cn = learn(dag, df, ImpreciseDirichletModel(); progress = false)
    @test cn isa CredalNetwork
    @test !isprecise(cn.nodes[findfirst(n -> n.name == :G, cn.nodes)])

    # :G has three parents, so its CPT carries one entry per state per configuration
    g = cn.nodes[findfirst(n -> n.name == :G, cn.nodes)]
    @test nrow(g.cpt.data) == 2 * 2 * 2 * 3

    # every column of the CPT is a coherent credal set, which is what `_verify_exhaustiveness` demands and what inference needs to enumerate extreme points at all
    for group in groupby(g.cpt.data, [:W, :R, :P])
        @test sum(i.lb for i in group.Π) <= 1 <= sum(i.ub for i in group.Π)
    end

    # the same structure fitted precisely stays precise and infers
    bn = learn(dag, df, PreciseDirichletModel(1); progress = false)
    @test bn isa BayesianNetwork
    order!(bn)
    @test infer(bn, :G, Evidence(:W => :warm)) isa EnhancedBayesianNetworks.Posterior

    # The end-to-end claim: a learned credal network is usable, not merely well formed. This is the only path reaching `_extreme_nodes` from learned CPTs, where a three-state node with a vacuous entry is the case that needed its vertices clamping to [0, 1]. Credal inference enumerates one network per combination of extreme points, so the node keeps three states but drops to two parents: at three it costs about 25 seconds against under two, for no extra coverage of the fitting path, which the assertions above already pin at three
    smaller = DirectAcyclicGraph()
    add_node!(smaller, :W)
    add_node!(smaller, :R; parents = [:W])
    add_node!(smaller, :G; parents = [:W, :R])
    credal = learn(smaller, df[!, [:W, :R, :G]], ImpreciseDirichletModel(); progress = false)
    order!(credal)
    @test infer(credal, :G, Evidence(:W => :warm)) isa EnhancedBayesianNetworks.CredalPosterior
end
