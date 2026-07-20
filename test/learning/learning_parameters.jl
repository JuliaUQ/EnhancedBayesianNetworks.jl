@testsnippet SetupLearningAsia begin
    # a DAG mirroring the SetupAsiaBN structure (fresh instance each call)
    builddag() = begin
        d = DirectAcyclicGraph()
        add_node!(d, :V)
        add_node!(d, :S)
        add_node!(d, :T, parents=[:V])
        add_node!(d, :L, parents=[:S])
        add_node!(d, :B, parents=[:S])
        add_node!(d, :E, parents=[:L, :T])
        add_node!(d, :D, parents=[:B, :E])
        add_node!(d, :X, parents=[:E])
        d
    end
    # well-sampled scenarios (avoids rare configs that are noise-dominated)
    recovery_scen = [
        (:V=>:YesV,),
        (:S=>:YesS,),
        (:V=>:YesV, :T=>:YesT),
        (:V=>:NoV, :T=>:YesT),
        (:S=>:YesS, :B=>:YesB),
        (:S=>:NoS, :B=>:YesB),
        (:S=>:YesS, :L=>:YesL),
        (:S=>:NoS, :L=>:YesL),
        (:B=>:YesB, :E=>:YesE, :D=>:YesD),
        (:B=>:NoB, :E=>:NoE, :D=>:YesD),
        (:E=>:YesE, :X=>:YesX),
        (:E=>:NoE, :X=>:YesX)
    ]
    # well-sampled configs (recover reliably even at the smaller N the slower EM test uses)
    common_scen = [
        (:V=>:YesV,), (:V=>:NoV,), (:S=>:YesS,), (:S=>:NoS,),
        (:V=>:NoV, :T=>:YesT),
        (:S=>:NoS, :B=>:YesB),
        (:S=>:NoS, :L=>:YesL),
        (:B=>:YesB, :E=>:NoE, :D=>:YesD),
        (:B=>:NoB, :E=>:NoE, :D=>:YesD),
        (:E=>:NoE, :X=>:YesX),
    ]
end

@testsnippet SetupLearningSimple begin
    # probability of a scenario in a network, e.g. getp(net, :V=>:yes, :T=>:t1)
    getp(net, sc...) = net.nodes[net.topology[last(sc).first]][sc...]

    # a small DAG (V -> T) with a complete dataset of known counts, so MLE/EM are exact:
    #   V: 8 :yes, 8 :no  |  T|V=:yes: 4 :t1, 4 :t2  |  T|V=:no: 2 :t1, 6 :t2
    learndata() = DataFrame(
        V=[fill(:yes, 8); fill(:no, 8)],
        T=[fill(:t1, 4); fill(:t2, 4); fill(:t1, 2); fill(:t2, 6)],
    )
    learndag() = (d=DirectAcyclicGraph(); add_node!(d, :V); add_node!(d, :T, parents=[:V]); d)
end

@testitem "DirectAcyclicGraph - Struct" setup=[ExtraDeps] begin
    dag = DirectAcyclicGraph()
    @test isa(dag, DirectAcyclicGraph)
    @test isempty(dag.nodes)
    @test dag.A == spzeros(Bool, 0, 0)

    add_node!(dag, :V)                       # root
    add_node!(dag, :S, [:extra])             # root, with an extra declared state
    add_node!(dag, :T, parents=[:V])         # one parent
    add_node!(dag, :E, parents=[:V, :S])     # two parents

    @test [n.name for n in dag.nodes] == [:V, :S, :T, :E]
    @test dag.topology == Dict(:V => 1, :S => 2, :T => 3, :E => 4)
    @test dag.A == sparse([1, 1, 2], [3, 4, 4], true, 4, 4)

    @test dag.states[:S] == [:extra]
    @test !haskey(dag.states, :V)

    @test parents(dag, :V) == Symbol[]
    @test parents(dag, :T) == [:V]
    @test parents(dag, :E) == [:V, :S]
    @test children(dag, :V) == [:T, :E]
    @test children(dag, :S) == [:E]
    @test children(dag, :E) == Symbol[]

    @test_throws ErrorException("Invalid DAG: node :V is already present") add_node!(dag, :V)
    @test_throws ErrorException("Invalid DAG: parent(s) [:Z] of :C are not defined; add them first") add_node!(dag, :C, parents=[:Z])
end

@testitem "DirectAcyclicGraph - add_node!" setup=[ExtraDeps] begin
    dag = DirectAcyclicGraph()

    # returns the dag; adds a root DiscreteNode with no edges and no stored states
    @test add_node!(dag, :V) === dag
    @test length(dag.nodes) == 1
    @test dag.nodes[1] isa DiscreteNode
    @test dag.nodes[1].name == :V
    @test dag.topology[:V] == 1
    @test dag.A == spzeros(Bool, 1, 1)
    @test !haskey(dag.states, :V)

    # extra declared states are stored; an empty states vector is treated as none
    add_node!(dag, :S, [:a, :b])
    @test dag.states[:S] == [:a, :b]
    add_node!(dag, :W, Symbol[])
    @test !haskey(dag.states, :W)

    # a parent wires an edge in A and is recorded as the node's CPT columns
    add_node!(dag, :T, parents=[:V])
    @test dag.topology[:T] == 4
    @test dag.A[dag.topology[:V], dag.topology[:T]]
    @test parents(dag, :T) == [:V]
    @test parents(dag.nodes[dag.topology[:T]]) == [:V]

    # multiple parents: one edge each, A grows by one row/col per node
    add_node!(dag, :E, parents=[:T, :S])
    @test size(dag.A) == (5, 5)
    @test parents(dag, :E) == [:S, :T]                        # adjacency: node-index order
    @test parents(dag.nodes[dag.topology[:E]]) == [:T, :S]    # CPT columns: declared order

    # errors
    @test_throws ErrorException("Invalid DAG: node :V is already present") add_node!(dag, :V)
    @test_throws ErrorException("Invalid DAG: parent(s) [:Z] of :C are not defined; add them first") add_node!(dag, :C, parents=[:Z])
end

@testitem "Parameters Learning - MLE" setup=[ExtraDeps, SetupAsiaBN, SetupLearningAsia, SetupLearningSimple] begin
    df = learndata()
    learned = learn_parameters_mle(learndag(), df)
    @test learned isa BayesianNetwork
    order!(learned)

    # exact maximum-likelihood estimates (count / total)
    @test getp(learned, :V=>:yes) == 0.5
    @test getp(learned, :V=>:no) == 0.5
    @test getp(learned, :V=>:yes, :T=>:t1) == 0.5
    @test getp(learned, :V=>:yes, :T=>:t2) == 0.5
    @test getp(learned, :V=>:no, :T=>:t1) == 0.25
    @test getp(learned, :V=>:no, :T=>:t2) == 0.75

    # Laplace smoothing: (count + alpha) / (total + alpha * k)
    smoothed = learn_parameters_mle(learndag(), df; alpha=1)
    @test getp(smoothed, :V=>:no, :T=>:t1) == (2 + 1) / (8 + 1 * 2)   # 0.3

    # a state declared on the DAG but never in the data stays at probability 0
    d1 = DirectAcyclicGraph();
    add_node!(d1, :T, [:t3])
    @test getp(learn_parameters_mle(d1, df), :T=>:t3) == 0.0

    # an unobserved parent configuration falls back to a uniform distribution
    d2 = DirectAcyclicGraph();
    add_node!(d2, :V, [:maybe]);
    add_node!(d2, :T, parents=[:V])
    fb = learn_parameters_mle(d2, df)
    @test getp(fb, :V=>:maybe) == 0.0
    @test getp(fb, :V=>:maybe, :T=>:t1) == 0.5
    @test getp(fb, :V=>:maybe, :T=>:t2) == 0.5

    # sample a large dataset from the known network, learn its parameters back, and check recovery.
    df = sample(bn, 50000)
    learned = learn_parameters_mle(builddag(), df)
    @test learned isa BayesianNetwork
    order!(learned)
    @test all(abs(getp(learned, sc...) - getp(bn, sc...)) < 0.06 for sc in recovery_scen)
end

@testitem "Parameters Learning - EM" setup=[ExtraDeps, SetupAsiaBN, SetupLearningAsia, SetupLearningSimple] begin
    # complete data: EM reduces exactly to MLE
    complete = learn_parameters_em(learndag(), allowmissing(learndata()))
    @test complete isa BayesianNetwork
    order!(complete)
    @test getp(complete, :V=>:yes) == 0.5
    @test getp(complete, :V=>:yes, :T=>:t1) == 0.5
    @test getp(complete, :V=>:no, :T=>:t1) == 0.25
    @test getp(complete, :V=>:no, :T=>:t2) == 0.75

    # T missing (V always observed): EM converges to the MLE over the observed-T rows
    dfm = DataFrame(
        V=[fill(:yes, 10); fill(:no, 10)],
        T=[fill(:t1, 4); fill(:t2, 4); [missing, missing]; fill(:t1, 2); fill(:t2, 6); [missing, missing]],
    )
    learned = learn_parameters_em(learndag(), dfm; tol=1e-10)
    @test learned isa BayesianNetwork
    order!(learned)
    @test getp(learned, :V=>:yes) == 0.5                              # V fully observed -> exact
    @test isapprox(getp(learned, :V=>:yes, :T=>:t1), 0.5; atol=1e-6)
    @test isapprox(getp(learned, :V=>:no, :T=>:t1), 0.25; atol=1e-6)
    @test isapprox(getp(learned, :V=>:no, :T=>:t2), 0.75; atol=1e-6)

    # sample from the known network, blank ~14% of cells deterministically, learn with EM, and
    # check recovery of the true CPTs on the well-sampled configurations. Unseeded on purpose.
    df = sample(bn, 3000)
    dfm = allowmissing(df)
    for (j, c) in enumerate(names(dfm)), i in 1:nrow(dfm)
        (i + j) % 7 == 0 && (dfm[i, c] = missing)
    end
    learned = learn_parameters_em(builddag(), dfm)
    @test learned isa BayesianNetwork
    order!(learned)
    @test all(abs(getp(learned, sc...) - getp(bn, sc...)) < 0.05 for sc in common_scen)

    # --- unit tests for the internal EM helpers (on a small V -> T graph) ---
    dag = (d=DirectAcyclicGraph(); add_node!(d, :V); add_node!(d, :T, parents=[:V]); d)
    domains = Dict(:V => [:no, :yes], :T => [:t1, :t2])

    # _em_uniform: every conditional distribution is 1/k
    u = EnhancedBayesianNetworks._em_uniform(dag, domains)
    @test u isa BayesianNetwork
    @test getp(u, :V=>:yes) == 0.5
    @test getp(u, :V=>:yes, :T=>:t1) == 0.5
    @test getp(u, :V=>:no, :T=>:t2) == 0.5

    # _em_estep: a complete row keeps weight 1; a T-missing row splits by P(T|V) under `net`.
    # net: P(V=yes)=0.5, P(T=t1|yes)=0.8 -> the missing row weights are 0.8 / 0.2
    Vn = DiscreteNode(:V);
    Vn[:V=>:yes]=0.5;
    Vn[:V=>:no]=0.5
    Tn = DiscreteNode(:T, [:V])
    Tn[:V=>:yes, :T=>:t1]=0.8;
    Tn[:V=>:yes, :T=>:t2]=0.2
    Tn[:V=>:no, :T=>:t1]=0.3;
    Tn[:V=>:no, :T=>:t2]=0.7
    net = BayesianNetwork([Vn, Tn]);
    add_child!(net, :V, :T);
    order!(net)
    edf = DataFrame(V=[:yes, :yes], T=[:t1, missing])
    completed = EnhancedBayesianNetworks._em_estep(dag, edf, net, domains)
    @test nrow(completed) == 3
    @test completed.V == [:yes, :yes, :yes]
    @test completed.T == [:t1, :t1, :t2]
    @test isapprox(completed.weight, [1.0, 0.8, 0.2])
    @test sum(completed.weight) ≈ 2.0

    # _em_mstep: weighted maximum likelihood (weights replace row counts)
    comp = DataFrame(V=[:yes, :yes, :no, :no], T=[:t1, :t2, :t1, :t2], weight=[3.0, 1.0, 1.0, 3.0])
    m = EnhancedBayesianNetworks._em_mstep(dag, comp, domains, 0)
    @test m isa BayesianNetwork
    @test getp(m, :V=>:yes) == 0.5
    @test getp(m, :V=>:yes, :T=>:t1) == 0.75
    @test getp(m, :V=>:no, :T=>:t1) == 0.25
    ms = EnhancedBayesianNetworks._em_mstep(dag, comp, domains, 1)
    @test getp(ms, :V=>:yes, :T=>:t1) == (3 + 1) / (4 + 1 * 2)   # Laplace: 0.6666...

    # _em_maxchange: largest CPT-entry difference between two networks (rows aligned)
    a = EnhancedBayesianNetworks._em_uniform(dag, domains)
    b = EnhancedBayesianNetworks._em_mstep(dag, comp, domains, 0)
    @test EnhancedBayesianNetworks._em_maxchange(a, a) == 0.0
    @test EnhancedBayesianNetworks._em_maxchange(a, b) == 0.25
end

@testitem "Parameters Learning - learn" setup=[ExtraDeps, SetupLearningSimple] begin
    df = learndata()

    # complete data -> MLE (exact MLE estimates)
    l = learn(learndag(), df)
    @test l isa BayesianNetwork
    order!(l)
    @test getp(l, :V=>:yes) == 0.5
    @test getp(l, :V=>:no, :T=>:t1) == 0.25
    @test getp(l, :V=>:no, :T=>:t2) == 0.75

    # keyword forwarding: alpha reaches the chosen algorithm
    @test getp(learn(learndag(), df; alpha=1), :V=>:no, :T=>:t1) == (2 + 1) / (8 + 1 * 2)   # 0.3

    # missing data -> EM (converges to the MLE over observed-T rows, not MLE-on-missing)
    dfm = DataFrame(
        V=[fill(:yes, 10); fill(:no, 10)],
        T=[fill(:t1, 4); fill(:t2, 4); [missing, missing]; fill(:t1, 2); fill(:t2, 6); [missing, missing]],
    )
    le = learn(learndag(), dfm; tol=1e-10)
    @test le isa BayesianNetwork
    order!(le)
    @test getp(le, :V=>:yes) == 0.5
    @test isapprox(getp(le, :V=>:no, :T=>:t1), 0.25; atol=1e-6)
end